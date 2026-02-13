import sys
from pathlib import Path
import time
from datetime import timedelta
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QMessageBox, 
    QDoubleSpinBox, QSpinBox, QCheckBox, QProgressBar, QSlider
)
from PySide6.QtCore import ( Qt, QThread, Signal, Slot )
from conversionHandling.conversion import convert_hdf5_to_omezarr
from conversionHandling.helpers.visualize import open_in_napari
from conversionHandling.helpers.storage import StorageType
from conversionHandling.helpers.sysinfo import detect_system

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


# -------------------- Worker Thread --------------------

class ConversionWorker(QThread):
    """Background thread for running conversion without blocking GUI"""
    progress_updated = Signal(int, int, int, int, float, float)  # block_count, total_blocks, rate, eta
    conversion_finished = Signal(Path, float)  # store_path, total_time
    conversion_failed = Signal(str)  # error_message
    
    def __init__(self, h5_path, out_path, chunks, mode, safety_factor, compression_level, storage, memory_limit_bytes, system):
        super().__init__()
        self.h5_path = h5_path
        self.out_path = out_path
        self.chunks = chunks
        self.mode = mode
        self.safety_factor = safety_factor
        self.compression_level = compression_level
        self.storage = storage
        self.memory_limit_bytes = memory_limit_bytes
        self.system = system
    
    def run(self):
        """Execute conversion in background thread"""
        try:
            start_time = time.time()
            store_path = convert_hdf5_to_omezarr(
                self.h5_path,
                self.out_path,
                target_chunks=self.chunks,
                mode=self.mode,
                safety_factor=self.safety_factor,
                compression_level=self.compression_level,
                storage=self.storage,
                memory_limit_bytes=self.memory_limit_bytes,
                system=self.system,
                progress_callback=self.handle_progress,
                downsample_factor=2
            )
            total_seconds = time.time() - start_time
            self.conversion_finished.emit(store_path, total_seconds)
        except Exception as e:
            self.conversion_failed.emit(str(e))
    
    def handle_progress(self, level, progress_levels, block_count, total_blocks, rate, eta):
        """Called by conversion function to report progress"""
        self.progress_updated.emit(level, progress_levels, block_count, total_blocks, rate, eta)

# -------------------- GUI --------------------

class ConverterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 → OME-Zarr Converter")
        self.setMinimumWidth(520)

        self.worker = None  # Keep reference to worker thread

        self.system = detect_system()
        available_gb = int(self.system.available_ram_gb)

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

        self.ram_slider = QSlider(Qt.Horizontal)
        self.ram_label = QLabel()
        self.ram_warning_label = QLabel()
        self.ram_warning_label.setStyleSheet("color: orange;")

        if available_gb < 4:
            # Allow conversion, but limit slider to actual RAM
            self.ram_slider.setMinimum(1)
            self.ram_slider.setMaximum(available_gb)
            self.ram_slider.setValue(available_gb)
            self.ram_warning_label.setText(
                "⚠ Low available RAM (<4GB). Conversion may be slow or unstable."
            )
            self.ram_warning_label.show()
        else:
            self.ram_slider.setMinimum(4) 
            self.ram_slider.setMaximum(available_gb)
            self.ram_slider.setValue(min(32, available_gb))
            self.ram_warning_label.hide()

        def update_ram_label(value):
            self.ram_label.setText(f"Of available system memory: {value} GB")

        self.ram_slider.valueChanged.connect(update_ram_label)
        update_ram_label(self.ram_slider.value())

        self.safety_factor_spin = QDoubleSpinBox()
        self.safety_factor_spin.setRange(0.1, 0.95)
        self.safety_factor_spin.setSingleStep(0.05)
        self.safety_factor_spin.setDecimals(2)
        self.safety_factor_spin.setValue(0.75)
        self.safety_factor_spin.setSuffix(" × of selected RAM")

        self.compression_spin = QSpinBox()
        self.compression_spin.setRange(1, 22)
        self.compression_spin.setValue(3)
        self.compression_spin.setToolTip("Zstd compression level (1 = fast, 22 = max)")

        self.storage_select = QComboBox()
        self.storage_select.addItem("HDD (Max 2 Workers)", StorageType.HDD)
        self.storage_select.addItem("SATA SSD (Max 4 Workers)", StorageType.SATA_SSD)
        self.storage_select.addItem("NVMe SSD (Max 8 Workers)", StorageType.NVME)

        self.storage_select.setCurrentIndex(2)  # default = NVMe SSD

        self.mode_select = QComboBox()
        self.mode_select.addItems(["Parallel", "Sequential"])

        self.run_button = QPushButton("Convert")
        self.run_button.clicked.connect(self.run_conversion)

        self.visualize_checkbox = QCheckBox("Open result in viewer")
        self.visualize_checkbox.setChecked(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)

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
        layout.addWidget(self.ram_label)
        layout.addWidget(self.ram_slider)
        layout.addWidget(self.ram_warning_label)

        safety_compression_hstack = QHBoxLayout()
        safety_vstack = QVBoxLayout()
        safety_vstack.addWidget(QLabel("Safety factor"))
        safety_vstack.addWidget(self.safety_factor_spin)

        compression_vstack = QVBoxLayout()
        compression_vstack.addWidget(QLabel("Compression level (zstd)"))
        compression_vstack.addWidget(self.compression_spin)

        safety_compression_hstack.addLayout(safety_vstack)
        safety_compression_hstack.addSpacing(10)
        safety_compression_hstack.addLayout(compression_vstack)

        layout.addSpacing(10)
        layout.addLayout(safety_compression_hstack)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Write mode"))
        layout.addWidget(self.mode_select)

        layout.addSpacing(10)
        layout.addWidget(self.visualize_checkbox)

        layout.addSpacing(20)
        layout.addWidget(self.run_button, alignment=Qt.AlignCenter)

        layout.addSpacing(10)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        
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

        if mb > 8:
            self.chunk_feedback.setText(
                f"⚠ Chunk size ≈ {mb:.2f} MB (> 8 MB). Consider smaller chunks."
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
        selected_ram_gb = self.ram_slider.value()
        memory_limit_bytes = selected_ram_gb * 1_000_000_000
        
        # Disable controls during conversion
        self.run_button.setEnabled(False)
        self.ram_slider.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start worker thread
        self.worker = ConversionWorker(
            self.h5_path,
            self.out_path,
            chunks,
            mode,
            self.safety_factor_spin.value(),
            self.compression_spin.value(),
            self.storage_select.currentData(),
            memory_limit_bytes,
            self.system
        )
        
        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.conversion_finished.connect(self.on_conversion_finished)
        self.worker.conversion_failed.connect(self.on_conversion_failed)
        
        # Start conversion
        self.worker.start()
    
    @Slot(int, int, float, float)
    def update_progress(self, level, progress_levels, block_count, total_blocks, rate, eta):
        """Update progress bar and label"""
        progress_pct = int((block_count / total_blocks) * 100)
        self.progress_bar.setValue(progress_pct)
        
        eta_str = str(timedelta(seconds=int(eta))) if eta > 0 else "calculating..."
        self.progress_label.setText(
            f"Building OME-Zarr level {level} of {progress_levels}\n"
            f"Block {block_count}/{total_blocks} ({progress_pct}%) • "
            f"{rate:.1f} blocks/s • ETA: {eta_str}"
        )
    
    @Slot(Path, float)
    def on_conversion_finished(self, store_path, total_seconds):
        """Handle successful conversion completion"""
        # Re-enable controls
        self.run_button.setEnabled(True)
        self.ram_slider.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        # Visualization
        if self.visualize_checkbox.isChecked():
            try:
                open_in_napari(store_path)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Visualization failed",
                    str(e)
                )

        # Success message
        chunks = parse_chunks(self.chunk_input.text())
        mode = self.mode_select.currentText()
        msg = (
            f"Output folder: {self.out_path}\n"
            f"Chunks: {chunks}\n"
            f"Mode: {mode}\n\n"
            f"Total runtime: {timedelta(seconds=int(total_seconds))}"
        )
        QMessageBox.information(self, "Conversion Complete", msg)
    
    @Slot(str)
    def on_conversion_failed(self, error_msg):
        """Handle conversion failure"""
        # Re-enable controls
        self.run_button.setEnabled(True)
        self.ram_slider.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        QMessageBox.critical(self, "Conversion failed", error_msg)



# -------------------- main --------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ConverterGUI()
    gui.show()
    sys.exit(app.exec())