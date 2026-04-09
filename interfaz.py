import sys
import os
import serial
import serial.tools.list_ports
import struct
import time
import csv
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QSizePolicy, QFrame, QGridLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont, QColor

import pyqtgraph as pg

# ── Constantes ────────────────────────────────────────────────
GRAVITY = 9.80665

# Clasificación NAR/TRA por impulso total (Ns)
NAR_CLASSES = [
    ("1/4A", 0,      0.3125),
    ("1/2A", 0.3125, 0.625),
    ("A",    0.625,  1.25),
    ("B",    1.25,   2.5),
    ("C",    2.5,    5.0),
    ("D",    5.0,    10.0),
    ("E",    10.0,   20.0),
    ("F",    20.0,   40.0),
    ("G",    40.0,   80.0),
    ("H",    80.0,   160.0),
    ("I",    160.0,  320.0),
    ("J",    320.0,  640.0),
    ("K",    640.0,  1280.0),
    ("L",    1280.0, 2560.0),
    ("M",    2560.0, 5120.0),
]

def clasificar_motor(impulso_ns):
    for letra, lo, hi in NAR_CLASSES:
        if lo <= impulso_ns < hi:
            pct = (impulso_ns - lo) / (hi - lo) * 100
            return letra, pct
    if impulso_ns >= 5120:
        return "M+", 100.0
    return "—", 0.0


# ── Hilo de lectura serial ─────────────────────────────────────
class SerialReader(QThread):
    # Emite: newtons, seq, rssi (rssi=-999 si no disponible en trama)
    data_received = pyqtSignal(float, int, int)

    def __init__(self, port, baudrate):
        super().__init__()
        self.port     = port
        self.baudrate = baudrate
        self.running  = False
        self.ser      = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.running = True
            buf = bytearray()
            while self.running:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if chunk:
                    buf.extend(chunk)
                # Paquete binario: float(4) + uint32(4) = 8 bytes
                while len(buf) >= 8:
                    newtons, seq = struct.unpack('<fI', buf[:8])
                    buf = buf[8:]
                    self.data_received.emit(float(newtons), int(seq), -999)
        except Exception as e:
            print(f"[SerialReader] {e}")

    def send_command(self, char):
        if self.ser and self.ser.is_open:
            self.ser.write(char.encode())

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()
        self.wait()


# ── Widgets auxiliares ─────────────────────────────────────────
class KpiCard(QFrame):
    """Tarjeta de un solo KPI con etiqueta y valor grande."""
    def __init__(self, label, unit, color="#00e5ff"):
        super().__init__()
        self.unit  = unit
        self.color = color
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            KpiCard {{
                background: #1a1a2e;
                border: 1px solid {color}44;
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)

        self.lbl_title = QLabel(label.upper())
        self.lbl_title.setStyleSheet(f"color: {color}; font-size: 10px; letter-spacing: 2px;")

        self.lbl_value = QLabel("—")
        self.lbl_value.setStyleSheet(f"color: #ffffff; font-size: 26px; font-weight: bold;")

        self.lbl_unit = QLabel(unit)
        self.lbl_unit.setStyleSheet(f"color: {color}88; font-size: 11px;")

        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_value)
        lay.addWidget(self.lbl_unit)

    def set_value(self, val, decimals=2):
        if isinstance(val, float):
            self.lbl_value.setText(f"{val:.{decimals}f}")
        else:
            self.lbl_value.setText(str(val))


# ── Dashboard principal ────────────────────────────────────────
class RocketDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Horus Space Lab — Telemetría de Banco de Pruebas")
        self.resize(1280, 820)

        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0d0d1a; }
            QLabel      { color: #cccccc; font-family: 'Courier New', monospace; }
            QPushButton {
                background-color: #1e1e2e; color: #dddddd;
                border-radius: 4px; padding: 8px 16px;
                font-weight: bold; font-family: 'Courier New', monospace;
                border: 1px solid #333355;
            }
            QPushButton:hover  { background-color: #2a2a3e; border-color: #5555aa; }
            QPushButton:pressed { background-color: #111122; }
            QComboBox {
                background-color: #1e1e2e; color: #dddddd;
                padding: 6px; border: 1px solid #333355;
                font-family: 'Courier New', monospace;
            }
            QComboBox QAbstractItemView { background: #1e1e2e; color: #ddd; }
        """)

        # Estado interno
        self.time_data    = []
        self.thrust_n     = []      # siempre en Newtons
        self.impulse_data = []      # impulso acumulado (Ns)
        self.rssi_data    = []
        self.lost_pkts    = 0
        self.last_seq     = -1
        self.start_ts     = None
        self.is_kg        = False

        self._build_ui()

    # ── Construcción de la UI ──────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vlay = QVBoxLayout(root)
        vlay.setContentsMargins(12, 12, 12, 12)
        vlay.setSpacing(10)

        # ── Cabecera ──
        hdr = QHBoxLayout()
        self.lbl_logo = QLabel()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pix = QPixmap(os.path.join(script_dir, "HorusSlogan.png"))
        if not pix.isNull():
            self.lbl_logo.setPixmap(pix.scaledToHeight(52, Qt.TransformationMode.SmoothTransformation))
        else:
            self.lbl_logo.setText("HORUS")
            self.lbl_logo.setStyleSheet("color:#00e5ff; font-size:22px; font-weight:bold;")

        lbl_sys = QLabel("SISTEMA DE TELEMETRÍA  //  BANCO DE PRUEBAS")
        lbl_sys.setStyleSheet("color:#5566aa; font-size:11px; letter-spacing:3px;")

        self.lbl_status = QLabel("● DESCONECTADO")
        self.lbl_status.setStyleSheet("color:#ff4444; font-size:11px; font-weight:bold;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        hdr.addWidget(self.lbl_logo)
        hdr.addSpacing(16)
        hdr.addWidget(lbl_sys)
        hdr.addStretch()
        hdr.addWidget(self.lbl_status)
        vlay.addLayout(hdr)

        # Línea separadora
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #222244;")
        vlay.addWidget(sep)

        # ── Controles ──
        ctrl = QHBoxLayout()
        self.port_combo = QComboBox()
        self._refresh_ports()

        self.btn_refresh = QPushButton("↺ PUERTOS")
        self.btn_refresh.clicked.connect(self._refresh_ports)

        self.btn_connect = QPushButton("⏚ CONECTAR")
        self.btn_connect.clicked.connect(self._toggle_connection)

        self.btn_unit = QPushButton("UNIDAD: N")
        self.btn_unit.setStyleSheet("background:#1a1a3a; color:#00e5ff; border-color:#00e5ff44;")
        self.btn_unit.clicked.connect(self._toggle_unit)

        self.btn_start = QPushButton("▶  START")
        self.btn_start.setStyleSheet("background:#0d2b0d; color:#00ff88; border-color:#00ff8844;")
        self.btn_start.clicked.connect(lambda: self._send_cmd('S'))

        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setStyleSheet("background:#2b0d0d; color:#ff4444; border-color:#ff444444;")
        self.btn_stop.clicked.connect(lambda: self._send_cmd('T'))

        self.btn_clear = QPushButton("⌫ LIMPIAR")
        self.btn_clear.clicked.connect(self._clear_data)

        self.btn_export = QPushButton("↓ EXPORTAR CSV")
        self.btn_export.clicked.connect(self._export_csv)

        for w in [self.port_combo, self.btn_refresh, self.btn_connect,
                  self.btn_unit, self.btn_start, self.btn_stop,
                  self.btn_clear, self.btn_export]:
            ctrl.addWidget(w)
        vlay.addLayout(ctrl)

        # ── KPI Cards ──
        kpi_row = QHBoxLayout()
        self.kpi_thrust  = KpiCard("Empuje actual",  "N",   "#00e5ff")
        self.kpi_max     = KpiCard("Empuje máximo",  "N",   "#ff9800")
        self.kpi_impulse = KpiCard("Impulso total",  "N·s", "#00ff88")
        self.kpi_class   = KpiCard("Clase NAR/TRA",  "",    "#ff4488")
        self.kpi_rssi    = KpiCard("RSSI",           "dBm", "#aa88ff")
        self.kpi_lost    = KpiCard("Pkts perdidos",  "",    "#ff6644")

        for card in [self.kpi_thrust, self.kpi_max, self.kpi_impulse,
                     self.kpi_class, self.kpi_rssi, self.kpi_lost]:
            kpi_row.addWidget(card)
        vlay.addLayout(kpi_row)

        # ── Gráficas ──
        graphs_row = QHBoxLayout()

        # Gráfica 1: Curva de empuje
        self.graph_thrust = pg.PlotWidget(title="")
        self._style_graph(self.graph_thrust, "Tiempo (s)", "Empuje (N)", "#00e5ff")
        self.curve_thrust = self.graph_thrust.plot(
            pen=pg.mkPen(color='#00e5ff', width=2),
            fillLevel=0, brush=pg.mkBrush(color=(0, 229, 255, 25))
        )

        # Gráfica 2: Impulso acumulado
        self.graph_impulse = pg.PlotWidget(title="")
        self._style_graph(self.graph_impulse, "Tiempo (s)", "Impulso acum. (N·s)", "#00ff88")
        self.curve_impulse = self.graph_impulse.plot(
            pen=pg.mkPen(color='#00ff88', width=2)
        )

        graphs_row.addWidget(self.graph_thrust,  stretch=3)
        graphs_row.addWidget(self.graph_impulse, stretch=2)
        vlay.addLayout(graphs_row)

        # ── Barra de estado inferior ──
        bot = QHBoxLayout()
        self.lbl_packets = QLabel("Paquetes recibidos: 0")
        self.lbl_packets.setStyleSheet("color:#445566; font-size:10px;")
        self.lbl_time = QLabel("t = 0.000 s")
        self.lbl_time.setStyleSheet("color:#445566; font-size:10px;")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight)
        bot.addWidget(self.lbl_packets)
        bot.addStretch()
        bot.addWidget(self.lbl_time)
        vlay.addLayout(bot)

    # ── Helpers de estilo ──────────────────────────────────────
    def _style_graph(self, gw, xlabel, ylabel, color):
        gw.setBackground('#0d0d1a')
        gw.showGrid(x=True, y=True, alpha=0.15)
        gw.setLabel('left',   ylabel, color=color,  size='10pt')
        gw.setLabel('bottom', xlabel, color='#445566', size='9pt')
        gw.getPlotItem().titleLabel.setText(
            f"<span style='color:{color};font-size:11pt;letter-spacing:2px'>"
            f"{ylabel.upper()}</span>"
        )
        for ax in ['left', 'bottom']:
            gw.getAxis(ax).setPen(pg.mkPen('#222244'))
            gw.getAxis(ax).setTextPen(pg.mkPen('#556688'))

    # ── Lógica de conexión ─────────────────────────────────────
    def _refresh_ports(self):
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports if ports else ["(sin puertos)"])

    def _toggle_connection(self):
        if hasattr(self, 'serial_thread') and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.btn_connect.setText("⏚ CONECTAR")
            self.btn_connect.setStyleSheet("")
            self.lbl_status.setText("● DESCONECTADO")
            self.lbl_status.setStyleSheet("color:#ff4444; font-size:11px; font-weight:bold;")
        else:
            port = self.port_combo.currentText()
            if not port or port.startswith("("):
                return
            self.serial_thread = SerialReader(port, 115200)
            self.serial_thread.data_received.connect(self._on_data)
            self.serial_thread.start()
            self.start_ts = time.time()
            self.btn_connect.setText("⏚ DESCONECTAR")
            self.btn_connect.setStyleSheet(
                "background:#1a2a1a; color:#00ff88; border-color:#00ff8844;")
            self.lbl_status.setText("● CONECTADO")
            self.lbl_status.setStyleSheet("color:#00ff88; font-size:11px; font-weight:bold;")

    # ── Recepción de datos ─────────────────────────────────────
    def _on_data(self, newtons, seq, rssi):
        if self.start_ts is None:
            self.start_ts = time.time()

        t = time.time() - self.start_ts

        # Detectar paquetes perdidos
        if self.last_seq >= 0 and seq > self.last_seq + 1:
            self.lost_pkts += seq - self.last_seq - 1
        self.last_seq = seq

        # Signo correcto
        newtons = abs(newtons)

        # Acumular
        self.time_data.append(t)
        self.thrust_n.append(newtons)

        # Impulso acumulado con trapecios
        if len(self.thrust_n) >= 2:
            dt = self.time_data[-1] - self.time_data[-2]
            prev = self.impulse_data[-1] if self.impulse_data else 0.0
            self.impulse_data.append(prev + 0.5 * (self.thrust_n[-1] + self.thrust_n[-2]) * dt)
        else:
            self.impulse_data.append(0.0)

        if rssi != -999:
            self.rssi_data.append(rssi)

        self._refresh_display(t, rssi)

    def _refresh_display(self, t_now, rssi):
        if not self.thrust_n:
            return

        factor = 1.0 / GRAVITY if self.is_kg else 1.0
        unit   = "kg" if self.is_kg else "N"
        i_unit = "kg·s" if self.is_kg else "N·s"

        thrust_now = self.thrust_n[-1]  * factor
        max_thrust = max(self.thrust_n) * factor
        impulso    = self.impulse_data[-1]
        clase, pct = clasificar_motor(impulso)

        # KPIs
        self.kpi_thrust.lbl_unit.setText(unit)
        self.kpi_max.lbl_unit.setText(unit)
        self.kpi_impulse.lbl_unit.setText(i_unit)

        self.kpi_thrust.set_value(thrust_now)
        self.kpi_max.set_value(max_thrust)
        self.kpi_impulse.set_value(impulso * factor)
        self.kpi_class.lbl_value.setText(
            f"{clase}" if clase == "—" else f"{clase}  ({pct:.0f}%)"
        )
        self.kpi_lost.set_value(self.lost_pkts, 0)

        if rssi != -999:
            self.kpi_rssi.set_value(float(rssi), 0)
        elif self.rssi_data:
            self.kpi_rssi.set_value(float(self.rssi_data[-1]), 0)

        # Gráficas
        t_arr = np.array(self.time_data)
        self.curve_thrust.setData(t_arr, np.array(self.thrust_n) * factor)
        self.curve_impulse.setData(t_arr, np.array(self.impulse_data) * factor)

        # Barra inferior
        self.lbl_packets.setText(f"Paquetes recibidos: {len(self.thrust_n)}")
        self.lbl_time.setText(f"t = {t_now:.3f} s")

    # ── Controles de prueba ────────────────────────────────────
    def _toggle_unit(self):
        self.is_kg = not self.is_kg
        lbl = "kg" if self.is_kg else "N"
        self.btn_unit.setText(f"UNIDAD: {lbl}")
        self.graph_thrust.setLabel('left',   f"Empuje ({lbl})")
        self.graph_impulse.setLabel('left',  f"Impulso acum. ({'kg·s' if self.is_kg else 'N·s'})")
        if self.thrust_n:
            self._refresh_display(self.time_data[-1], -999)

    def _send_cmd(self, char):
        if hasattr(self, 'serial_thread') and self.serial_thread.isRunning():
            self.serial_thread.send_command(char)

    def _clear_data(self):
        self.time_data.clear()
        self.thrust_n.clear()
        self.impulse_data.clear()
        self.rssi_data.clear()
        self.lost_pkts = 0
        self.last_seq  = -1
        self.start_ts  = None
        self.curve_thrust.setData([], [])
        self.curve_impulse.setData([], [])
        for card in [self.kpi_thrust, self.kpi_max, self.kpi_impulse,
                     self.kpi_class, self.kpi_rssi, self.kpi_lost]:
            card.lbl_value.setText("—")
        self.lbl_packets.setText("Paquetes recibidos: 0")
        self.lbl_time.setText("t = 0.000 s")

    def _export_csv(self):
        if not self.thrust_n:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar telemetría", "", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Tiempo (s)", "Empuje (N)", "Impulso acum (Ns)",
                        "RSSI (dBm)", "Pkts perdidos"])
            for i in range(len(self.time_data)):
                rssi_val = self.rssi_data[i] if i < len(self.rssi_data) else ""
                w.writerow([
                    round(self.time_data[i], 4),
                    round(self.thrust_n[i], 4),
                    round(self.impulse_data[i], 4),
                    rssi_val,
                    self.lost_pkts
                ])


# ── Punto de entrada ───────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Courier New", 9))
    win = RocketDashboard()
    win.show()
    sys.exit(app.exec())