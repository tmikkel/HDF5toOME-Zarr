import sys
import math
from pathlib import Path
import time
from datetime import timedelta
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QMessageBox, 
    QDoubleSpinBox, QSpinBox
)
from PySide6.QtCore import Qt
from conversionHandling.conversion import convert_hdf5_to_omezarr
from conversionHandling.helpers.storage import StorageType

# -------------------- helpers --------------------

def parse_chunks(text: str):
    """Parse chunk string like '64,64,64' -> (64,64,64)"""
    try:
        parts = [int(p.strip()) for p in text.split(',')]
        if len(parts) != 3 or any(p <= 0 for p in parts):
            raise ValueError
        return tuple(parts)
    except Exception:
        return None


def chunk_bytes(chunks, dtype_bytes=4):
    """Estimate bytes per chunk (default float32 = 4 bytes)."""
    z, y, x = chunks
    return z * y * x * dtype_bytes


# -------------------- GUI --------------------

class ConverterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 → OME-Zarr Converter")
        self.setMinimumWidth(520)

        # ---- widgets ----
        self.h5_label = QLabel("No HDF5 file selected")
        self.h5_button = QPushButton("Select .h5 file")
        self.h5_button.clicked.connect(self.select_h5)

        self.out_label = QLabel("No output folder selected")
        self.out_button = QPushButton("Select output folder")
        self.out_button.clicked.connect(self.select_output)

        self.chunk_input = QLineEdit("64,64,64")
        self.chunk_feedback = QLabel("")
        self.chunk_feedback.setStyleSheet("color: orange")

        self.safety_factor_spin = QDoubleSpinBox()
        self.safety_factor_spin.setRange(0.1, 0.95)
        self.safety_factor_spin.setSingleStep(0.05)
        self.safety_factor_spin.setDecimals(2)
        self.safety_factor_spin.setValue(0.75)
        self.safety_factor_spin.setSuffix(" × RAM")

        self.compression_spin = QSpinBox()
        self.compression_spin.setRange(1, 22)
        self.compression_spin.setValue(3)
        self.compression_spin.setToolTip("Zstd compression level (1 = fast, 22 = max)")

        self.storage_select = QComboBox()
        self.storage_select.addItem("HDD", StorageType.HDD)
        self.storage_select.addItem("SATA SSD", StorageType.SATA_SSD)
        self.storage_select.addItem("NVMe SSD", StorageType.NVME)

        self.storage_select.setCurrentIndex(2)  # default = NVMe SSD

        self.mode_select = QComboBox()
        self.mode_select.addItems(["Sequential", "Parallel (Dask)"])

        self.run_button = QPushButton("Convert")
        self.run_button.clicked.connect(self.run_conversion)

        # ---- layouts ----
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Input HDF5 file"))
        layout.addWidget(self.h5_label)
        layout.addWidget(self.h5_button)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Output folder"))
        layout.addWidget(self.out_label)
        layout.addWidget(self.out_button)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Target chunk size (z,y,x)"))
        layout.addWidget(self.chunk_input)
        layout.addWidget(self.chunk_feedback)
        self.chunk_input.textChanged.connect(self.check_chunks)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Storage type"))
        layout.addWidget(self.storage_select)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Safety factor"))
        layout.addWidget(self.safety_factor_spin)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Compression level (zstd)"))
        layout.addWidget(self.compression_spin)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Write mode"))
        layout.addWidget(self.mode_select)

        layout.addSpacing(20)
        layout.addWidget(self.run_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

        self.h5_path: Path | None = None
        self.out_path: Path | None = None

        self.check_chunks()

    # -------------------- callbacks --------------------

    def select_h5(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select HDF5 file", "", "HDF5 files (*.h5 *.hdf5)")
        if path:
            self.h5_path = Path(path)
            self.h5_label.setText(str(self.h5_path))

    def select_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.out_path = Path(path)
            self.out_label.setText(str(self.out_path))

    def check_chunks(self):
        chunks = parse_chunks(self.chunk_input.text())
        if not chunks:
            self.chunk_feedback.setText("Invalid chunk format. Use e.g. 64,64,64")
            return

        bytes_per_chunk = chunk_bytes(chunks)
        mb = bytes_per_chunk / (1024**2)

        if mb > 4:
            self.chunk_feedback.setText(
                f"⚠ Chunk size ≈ {mb:.2f} MB (> 2 MB). Consider smaller chunks."
            )
        else:
            self.chunk_feedback.setText(f"✓ Chunk size ≈ {mb:.2f} MB")

    def run_conversion(self):
        if not self.h5_path or not self.h5_path.exists():
            QMessageBox.critical(self, "Error", "Please select a valid HDF5 file")
            return
        if not self.out_path or not self.out_path.exists():
            QMessageBox.critical(self, "Error", "Please select an output folder")
            return

        chunks = parse_chunks(self.chunk_input.text())
        if not chunks:
            QMessageBox.critical(self, "Error", "Invalid chunk size")
            return

        mode = self.mode_select.currentText()
        
        # ---- run conversion ----
        try:
            start_time = time.time()
            convert_hdf5_to_omezarr(
                self.h5_path,
                self.out_path,
                target_chunks=chunks,
                mode=mode,
                safety_factor = self.safety_factor_spin.value(),
                compression_level = self.compression_spin.value(),
                storage = self.storage_select.currentData()  # StorageType enum
            )
            total_seconds = time.time() - start_time
        except Exception as e:
            QMessageBox.critical(self, "Conversion failed", str(e))
            return

        msg = (
            f"Output folder: {self.out_path}\n"
            f"Chunks: {chunks}\n"
            f"Mode: {mode}\n\n"
            f"  Total runtime: {timedelta(seconds=int(total_seconds))}"
        )

        QMessageBox.information(self, "Conversion", msg)


# -------------------- main --------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ConverterGUI()
    gui.show()
    sys.exit(app.exec())