import sys
import os
import serial
import serial.tools.list_ports
import struct
import time
import csv
import json
import numpy as np
import glob

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QFrame
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont, QIcon

import pyqtgraph as pg

# --- CONFIGURACIÓN DE RUTAS PARA EJECUTABLE ---
def resource_path(relative_path):
    """ Gestiona las rutas de archivos para que funcionen en el .exe / ejecutable """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

GRAVITY          = 9.80665
THRUST_THRESHOLD = 2.0
MODE_CABLE       = "cable"
MODE_LORA        = "lora"
CONFIG_FILE      = os.path.join(os.path.expanduser("~"), ".horus_config.json")

NAR_CLASSES = [
    ("A",  1.26,       2.5),    ("B",  2.5,        5.0),
    ("C",  5.0,        10.0),   ("D",  10.0,       20.0),
    ("E",  20.0,       40.0),   ("F",  40.0,       80.0),
    ("G",  80.0,       160.0),  ("H",  160.0,      320.0),
    ("I",  320.0,      640.0),  ("J",  640.0,      1280.0),
    ("K",  1280.0,     2560.0), ("L",  2560.0,     5120.0),
    ("M",  5120.0,     10240.0),("N",  10240.0,    20480.0),
    ("O",  20480.0,    40960.0),("P",  40960.0,    81920.0),
    ("Q",  81920.0,    163840.0),("R", 163840.0,   327680.0),
    ("S",  327680.0,   655360.0),("T", 655360.0,   1310720.0),
    ("U",  1310720.0,  2621440.0),("V",2621440.0,  5242880.0),
]

def clasificar_motor(impulso_ns):
    for letra, lo, hi in NAR_CLASSES:
        if lo <= impulso_ns < hi:
            return letra, (impulso_ns - lo) / (hi - lo) * 100
    if impulso_ns >= 5242880:
        return "V+", 100.0
    return "—", 0.0

def cargar_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def guardar_config(data):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass

class AutoSaver(QThread):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.queue    = []
        self.running  = True
        self._write_header()

    def _write_header(self):
        with open(self.filepath, 'w', newline='') as f:
            csv.writer(f).writerow([
                "Tiempo (s)", "Empuje (N)", "Empuje (kg)",
                "Impulso acum (Ns)", "Impulso acum (kgs)"
            ])

    def add_row(self, t, n, imp):
        self.queue.append((t, n, imp))

    def run(self):
        while self.running:
            if self.queue:
                rows = self.queue.copy()
                self.queue.clear()
                with open(self.filepath, 'a', newline='') as f:
                    w = csv.writer(f)
                    for t, n, imp in rows:
                        w.writerow([
                            round(t,   4), round(n,   4), round(n   / GRAVITY, 4),
                            round(imp, 4), round(imp / GRAVITY, 4),
                        ])
            time.sleep(0.5)

    def stop(self):
        self.running = False
        self.wait()

class SerialReader(QThread):
    data_received = pyqtSignal(float, int, int)
    disconnected  = pyqtSignal()
    PACKET_SIZE   = {MODE_CABLE: 8, MODE_LORA: 9}

    def __init__(self, port, baudrate, mode=MODE_CABLE):
        super().__init__()
        self.port     = port
        self.baudrate = baudrate
        self.mode     = mode
        self.running  = False
        self.ser      = None

    def run(self):
        pkt_size = self.PACKET_SIZE[self.mode]
        try:
            self.ser     = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.running = True
            buf          = bytearray()
            while self.running:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if chunk:
                    buf.extend(chunk)
                while len(buf) >= pkt_size:
                    raw = buf[:pkt_size]
                    buf = buf[pkt_size:]
                    if self.mode == MODE_CABLE:
                        newtons, seq = struct.unpack('<fI', raw)
                        rssi = -999
                    else:
                        newtons, seq, rssi = struct.unpack('<fIb', raw)
                    if -500.0 < newtons < 5000.0 and seq < 1_000_000:
                        self.data_received.emit(float(newtons), int(seq), int(rssi))
        except Exception as e:
            print(f"[SerialReader] {e}")
        finally:
            self.disconnected.emit()

    def send_command(self, char):
        if self.ser and self.ser.is_open:
            self.ser.write(char.encode())

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.wait()

class KpiCard(QFrame):
    def __init__(self, label, unit, color="#00e5ff"):
        super().__init__()
        self.color = color
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"KpiCard{{background:#1a1a2e;border:1px solid {color}44;border-radius:8px;}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        self.lbl_title = QLabel(label.upper())
        self.lbl_title.setStyleSheet(f"color:{color};font-size:10px;letter-spacing:2px;")
        self.lbl_value = QLabel("—")
        self.lbl_value.setStyleSheet("color:#ffffff;font-size:24px;font-weight:bold;")
        self.lbl_unit  = QLabel(unit)
        self.lbl_unit.setStyleSheet(f"color:{color}88;font-size:11px;")
        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_value)
        lay.addWidget(self.lbl_unit)

    def set_value(self, val, decimals=2):
        self.lbl_value.setText(f"{val:.{decimals}f}" if isinstance(val,(float,int)) else str(val))

    def reset(self):
        self.lbl_value.setText("—")

class RocketDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Horus Space Lab — Telemetría de Banco de Pruebas")
        self.resize(1280, 860)
        
        # Cargar Icono de Ventana
        self.setWindowIcon(QIcon(resource_path("favicon.ico")))
        
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#0d0d1a;}
            QLabel{color:#cccccc;font-family:'Courier New',monospace;}
            QPushButton{background:#1e1e2e;color:#dddddd;border-radius:4px;
                padding:8px 16px;font-weight:bold;font-family:'Courier New',monospace;
                border:1px solid #333355;}
            QPushButton:hover{background:#2a2a3e;border-color:#5555aa;}
            QPushButton:pressed{background:#111122;}
            QComboBox{background:#1e1e2e;color:#dddddd;padding:6px;
                border:1px solid #333355;font-family:'Courier New',monospace;}
        """)

        self.time_data    = []
        self.thrust_n     = []
        self.impulse_data = []
        self.rssi_data    = []
        self.lost_pkts    = 0
        self.last_seq     = -1
        self.start_ts     = None
        self.is_kg        = False
        self.tare_offset  = 0.0
        self.conn_mode    = MODE_LORA
        self.auto_saver   = None
        self.armed        = False

        self.reconnect_timer = QTimer()
        self.reconnect_timer.setInterval(3000)
        self.reconnect_timer.timeout.connect(self._try_reconnect)

        self.config = cargar_config()
        self._build_ui()

        if self.config.get("last_port"):
            QTimer.singleShot(800, self._auto_reconnect)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vlay = QVBoxLayout(root)
        vlay.setContentsMargins(12, 12, 12, 12)
        vlay.setSpacing(8)

        hdr = QHBoxLayout()
        self.lbl_logo = QLabel()
        pix  = QPixmap(resource_path("HorusSlogan.png"))
        if not pix.isNull():
            self.lbl_logo.setPixmap(pix.scaledToHeight(52, Qt.TransformationMode.SmoothTransformation))
        else:
            self.lbl_logo.setText("HORUS SPACE LAB")
            self.lbl_logo.setStyleSheet("color:#00e5ff;font-size:22px;font-weight:bold;")
        
        lbl_sys = QLabel("SISTEMA DE TELEMETRÍA  //  BANCO DE PRUEBAS")
        lbl_sys.setStyleSheet("color:#5566aa;font-size:11px;letter-spacing:3px;")
        self.lbl_status = QLabel("● DESCONECTADO")
        self.lbl_status.setStyleSheet("color:#ff4444;font-size:11px;font-weight:bold;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hdr.addWidget(self.lbl_logo); hdr.addSpacing(16)
        hdr.addWidget(lbl_sys); hdr.addStretch(); hdr.addWidget(self.lbl_status)
        vlay.addLayout(hdr)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#222244;"); vlay.addWidget(sep)

        mode_row = QHBoxLayout()
        self.btn_mode_cable = QPushButton("CABLE / RS-485")
        self.btn_mode_lora  = QPushButton("LoRa")
        self.btn_mode_cable.clicked.connect(lambda: self._set_mode(MODE_CABLE))
        self.btn_mode_lora.clicked.connect( lambda: self._set_mode(MODE_LORA))
        self.lbl_mode_badge = QLabel()
        mode_row.addWidget(QLabel("MODO:"))
        mode_row.addWidget(self.btn_mode_cable)
        mode_row.addWidget(self.btn_mode_lora)
        mode_row.addStretch()
        mode_row.addWidget(self.lbl_mode_badge)
        vlay.addLayout(mode_row)

        ctrl = QHBoxLayout()
        self.port_combo = QComboBox()
        self._refresh_ports()
        self.btn_refresh = QPushButton("↺")
        self.btn_refresh.setFixedWidth(36)
        self.btn_refresh.clicked.connect(self._refresh_ports)

        self.btn_connect = QPushButton("⏚ CONECTAR")
        self.btn_connect.clicked.connect(self._toggle_connection)

        self.btn_start = QPushButton("▶ START")
        self.btn_start.setStyleSheet("background:#0d2b0d;color:#00ff88;border-color:#00ff8844;")
        self.btn_start.clicked.connect(lambda: self._send_cmd('S'))

        self.btn_stop = QPushButton("■ STOP")
        self.btn_stop.setStyleSheet("background:#2b0d0d;color:#ff4444;border-color:#ff444444;")
        self.btn_stop.clicked.connect(lambda: self._send_cmd('T'))

        self.btn_unit = QPushButton("UNIDAD: N")
        self.btn_unit.setStyleSheet("background:#1a1a3a;color:#00e5ff;border-color:#00e5ff44;")
        self.btn_unit.clicked.connect(self._toggle_unit)

        self.btn_tare = QPushButton("⊙ TARAR")
        self.btn_tare.clicked.connect(self._tare)

        self.btn_clear = QPushButton("⌫ LIMPIAR")
        self.btn_clear.clicked.connect(self._clear_data)

        self.btn_export = QPushButton("↓ CSV")
        self.btn_export.clicked.connect(self._export_csv)

        for w in [self.port_combo, self.btn_refresh, self.btn_connect,
                  self.btn_start, self.btn_stop, self.btn_unit,
                  self.btn_tare, self.btn_clear, self.btn_export]:
            ctrl.addWidget(w)
        vlay.addLayout(ctrl)

        ign_row = QHBoxLayout()
        self.btn_arm = QPushButton("ARM")
        self.btn_arm.setStyleSheet("background:#2b1a00;color:#ffaa00;border:2px solid #ffaa0066;padding:10px 24px;")
        self.btn_arm.clicked.connect(self._toggle_arm)

        self.btn_fire = QPushButton("FIRE")
        self.btn_fire.setStyleSheet("background:#1a0000;color:#444444;border:2px solid #44444444;padding:10px 24px;")
        self.btn_fire.setEnabled(False)
        self.btn_fire.clicked.connect(self._fire)

        self.lbl_arm_status = QLabel("SISTEMA: SAFE")
        ign_row.addWidget(self.btn_arm); ign_row.addWidget(self.btn_fire)
        ign_row.addSpacing(16); ign_row.addWidget(self.lbl_arm_status)
        ign_row.addStretch()
        vlay.addLayout(ign_row)

        save_row = QHBoxLayout()
        self.lbl_savepath = QLabel("Autoguardado: (no iniciado)")
        save_row.addWidget(self.lbl_savepath)
        vlay.addLayout(save_row)

        kpi_row = QHBoxLayout()
        self.kpi_thrust  = KpiCard("Empuje actual",  "N")
        self.kpi_max     = KpiCard("Empuje máximo",  "N",    "#ff9800")
        self.kpi_impulse = KpiCard("Impulso total",  "N·s",  "#00ff88")
        self.kpi_class   = KpiCard("Clase NAR",      "",     "#ff4488")
        self.kpi_rssi    = KpiCard("RSSI",           "dBm",  "#aa88ff")
        self.kpi_lost    = KpiCard("Pkts perdidos",  "",     "#ff6644")
        for c in [self.kpi_thrust, self.kpi_max, self.kpi_impulse,
                  self.kpi_class, self.kpi_rssi, self.kpi_lost]:
            kpi_row.addWidget(c)
        vlay.addLayout(kpi_row)

        gr = QHBoxLayout()
        self.graph_thrust  = pg.PlotWidget()
        self.graph_impulse = pg.PlotWidget()
        self._style_graph(self.graph_thrust,  "Tiempo (s)", "Empuje (N)",    "#00e5ff")
        self._style_graph(self.graph_impulse, "Tiempo (s)", "Impulso (N·s)", "#00ff88")
        self.curve_thrust  = self.graph_thrust.plot( pen=pg.mkPen('#00e5ff', width=2))
        self.curve_impulse = self.graph_impulse.plot(pen=pg.mkPen('#00ff88', width=2))
        gr.addWidget(self.graph_thrust, 3); gr.addWidget(self.graph_impulse, 2)
        vlay.addLayout(gr)

        self._set_mode(MODE_LORA)

    def _style_graph(self, gw, xlabel, ylabel, color):
        gw.setBackground('#0d0d1a')
        gw.showGrid(x=True, y=True, alpha=0.15)
        gw.setLabel('left', ylabel, color=color)
        gw.setLabel('bottom', xlabel, color='#445566')

    def _refresh_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports if ports else ["No Ports"])

    def _set_mode(self, mode):
        self.conn_mode = mode
        self.lbl_mode_badge.setText("CABLE/RS-485" if mode == MODE_CABLE else "LoRa")
        
        # Lógica de Visibilidad solicitada
        is_lora = (mode == MODE_LORA)
        self.btn_start.setVisible(is_lora)
        self.btn_stop.setVisible(is_lora)
        self.kpi_rssi.setVisible(is_lora)
        
        # Opcional: También podrías ocultar ignición en cable si lo deseas
        # self.btn_arm.setVisible(is_lora)
        # self.btn_fire.setVisible(is_lora)

        if self.armed: self._disarm()

    def _toggle_connection(self):
        if hasattr(self, 'serial_thread') and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self._on_disconnected(intentional=True)
        else:
            self._connect()

    def _connect(self):
        port = self.port_combo.currentText()
        if not port or "No Ports" in port: return
        self.serial_thread = SerialReader(port, 115200, mode=self.conn_mode)
        self.serial_thread.data_received.connect(self._on_data)
        self.serial_thread.disconnected.connect(lambda: self._on_disconnected(intentional=False))
        self.serial_thread.start()
        self.lbl_status.setText("● CONECTADO")
        self.lbl_status.setStyleSheet("color:#00ff88;font-size:11px;font-weight:bold;")
        self._iniciar_autoguardado()

    def _on_disconnected(self, intentional=False):
        self.lbl_status.setText("● DESCONECTADO")
        self.lbl_status.setStyleSheet("color:#ff4444;font-size:11px;font-weight:bold;")
        if not intentional: self.reconnect_timer.start()

    def _try_reconnect(self):
        self.reconnect_timer.stop()
        self._connect()

    def _auto_reconnect(self):
        self._connect()

    def _send_cmd(self, char):
        if hasattr(self, 'serial_thread') and self.serial_thread.isRunning():
            self.serial_thread.send_command(char)

    def _iniciar_autoguardado(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(os.path.expanduser("~"), f"telemetria_{ts}.csv")
        self.auto_saver = AutoSaver(filepath)
        self.auto_saver.start()
        self.lbl_savepath.setText(f"Autoguardando: {filepath}")

    def _on_data(self, newtons, seq, rssi):
        if self.start_ts is None: self.start_ts = time.time()
        t = time.time() - self.start_ts
        newtons = abs(newtons) - self.tare_offset
        thrust_clean = newtons if newtons >= THRUST_THRESHOLD else 0.0
        self.time_data.append(t)
        self.thrust_n.append(thrust_clean)
        
        if len(self.thrust_n) >= 2:
            dt = self.time_data[-1] - self.time_data[-2]
            prev_imp = self.impulse_data[-1] if self.impulse_data else 0.0
            self.impulse_data.append(prev_imp + thrust_clean * dt)
        else:
            self.impulse_data.append(0.0)

        if self.auto_saver: self.auto_saver.add_row(t, thrust_clean, self.impulse_data[-1])
        self._refresh_display(t, rssi)

    def _refresh_display(self, t_now, rssi):
        if not self.thrust_n: return
        f = 1.0 / GRAVITY if self.is_kg else 1.0
        self.kpi_thrust.set_value(self.thrust_n[-1] * f)
        self.kpi_max.set_value(max(self.thrust_n) * f)
        self.kpi_impulse.set_value(self.impulse_data[-1] * f)
        clase, pct = clasificar_motor(self.impulse_data[-1])
        self.kpi_class.lbl_value.setText(f"{clase} ({pct:.0f}%)")
        if rssi != -999: self.kpi_rssi.set_value(float(rssi), 0)
        self.curve_thrust.setData(self.time_data, [x*f for x in self.thrust_n])

    def _toggle_arm(self):
        self.armed = not self.armed
        if self.armed:
            self.btn_fire.setEnabled(True)
            self.btn_fire.setStyleSheet("background:#4a0000;color:#ff2222;border:2px solid #ff2222;")
            self.lbl_arm_status.setText("SISTEMA: ARMADO")
        else:
            self._disarm()

    def _disarm(self):
        self.armed = False
        self.btn_fire.setEnabled(False)
        self.btn_fire.setStyleSheet("background:#1a0000;color:#444444;border:2px solid #44444444;")
        self.lbl_arm_status.setText("SISTEMA: SAFE")

    def _fire(self):
        if self.armed:
            self._send_cmd('F')
            self._send_cmd('S')
            self._disarm()

    def _tare(self):
        if len(self.thrust_n) > 5:
            self.tare_offset = float(np.mean(self.thrust_n[-10:]))

    def _clear_data(self):
        self.time_data.clear(); self.thrust_n.clear(); self.impulse_data.clear()
        self.curve_thrust.setData([], []); self.start_ts = None

    def _toggle_unit(self):
        self.is_kg = not self.is_kg
        self.btn_unit.setText(f"UNIDAD: {'kg' if self.is_kg else 'N'}")

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Exportar CSV", "", "CSV (*.csv)")
        if path:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["T", "N", "KG"])
                for i in range(len(self.time_data)):
                    w.writerow([self.time_data[i], self.thrust_n[i], self.thrust_n[i]/GRAVITY])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RocketDashboard()
    win.show()
    sys.exit(app.exec())