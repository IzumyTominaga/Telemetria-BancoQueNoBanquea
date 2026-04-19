import sys
import os
import serial
import serial.tools.list_ports
import struct
import time
import csv
import numpy as np
import glob

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QSizePolicy, QFrame, QGridLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont, QColor

import pyqtgraph as pg

# ─────────────────────────────────────────────
GRAVITY          = 9.80665
THRUST_THRESHOLD = 2.0    # N  — ajustar según ruido real de la celda
ACTIVE_SAMPLES   = 3      # muestras consecutivas sobre umbral para "disparo"
TARE_SAMPLES     = 30     # cuántas muestras usar para calcular la tara

# Modos de conexión
MODE_CABLE = "cable"   # RS-485  — paquete 8 bytes: float + uint32
MODE_LORA  = "lora"    # LoRa    — paquete 9 bytes: float + uint32 + int8(RSSI)

NAR_CLASSES = [
    ("A",  1.26,        2.5),
    ("B",  2.5,         5.0),
    ("C",  5.0,         10.0),
    ("D",  10.0,        20.0),
    ("E",  20.0,        40.0),
    ("F",  40.0,        80.0),
    ("G",  80.0,        160.0),
    ("H",  160.0,       320.0),
    ("I",  320.0,       640.0),
    ("J",  640.0,       1280.0),
    ("K",  1280.0,      2560.0),
    ("L",  2560.0,      5120.0),
    ("M",  5120.0,      10240.0),
    ("N",  10240.0,     20480.0),
    ("O",  20480.0,     40960.0),
    ("P",  40960.0,     81920.0),
    ("Q",  81920.0,     163840.0),
    ("R",  163840.0,    327680.0),
    ("S",  327680.0,    655360.0),
    ("T",  655360.0,    1310720.0),
    ("U",  1310720.0,   2621440.0),
    ("V",  2621440.0,   5242880.0),
]

def clasificar_motor(impulso_ns):
    for letra, lo, hi in NAR_CLASSES:
        if lo <= impulso_ns < hi:
            pct = (impulso_ns - lo) / (hi - lo) * 100
            return letra, pct
    if impulso_ns >= 5242880:
        return "V+", 100.0
    return "—", 0.0


# ─────────────────────────────────────────────
class SerialReader(QThread):
    """
    Lector serie genérico.
    Emite: (newtons: float, seq: int, rssi: int)
    rssi = -999 cuando el modo cable no envía RSSI.
    """
    data_received = pyqtSignal(float, int, int)

    # Tamaño de paquete según modo
    PACKET_SIZE = {MODE_CABLE: 8, MODE_LORA: 9}

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
                    raw      = buf[:pkt_size]
                    buf      = buf[pkt_size:]

                    if self.mode == MODE_CABLE:
                        newtons, seq = struct.unpack('<fI', raw)
                        rssi         = -999
                    else:  # LoRa — 9 bytes: float + uint32 + int8
                        newtons, seq, rssi = struct.unpack('<fIb', raw)

                    # Sanidad básica
                    if -500.0 < newtons < 5000.0 and seq < 1_000_000:
                        self.data_received.emit(float(newtons), int(seq), int(rssi))

        except Exception as e:
            print(f"[SerialReader] {e}")

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.wait()


# ─────────────────────────────────────────────
class KpiCard(QFrame):
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
        self.lbl_title.setStyleSheet(
            f"color: {color}; font-size: 10px; letter-spacing: 2px;")

        self.lbl_value = QLabel("—")
        self.lbl_value.setStyleSheet(
            "color: #ffffff; font-size: 26px; font-weight: bold;")

        self.lbl_unit = QLabel(unit)
        self.lbl_unit.setStyleSheet(
            f"color: {color}88; font-size: 11px;")

        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_value)
        lay.addWidget(self.lbl_unit)

    def set_value(self, val, decimals=2):
        if isinstance(val, float):
            self.lbl_value.setText(f"{val:.{decimals}f}")
        else:
            self.lbl_value.setText(str(val))


# ─────────────────────────────────────────────
class RocketDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Horus Space Lab — Telemetría de Banco de Pruebas")
        self.resize(1280, 860)

        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0d0d1a; }
            QLabel      { color: #cccccc; font-family: 'Courier New', monospace; }
            QPushButton {
                background-color: #1e1e2e; color: #dddddd;
                border-radius: 4px; padding: 8px 16px;
                font-weight: bold; font-family: 'Courier New', monospace;
                border: 1px solid #333355;
            }
            QPushButton:hover   { background-color: #2a2a3e; border-color: #5555aa; }
            QPushButton:pressed { background-color: #111122; }
            QComboBox {
                background-color: #1e1e2e; color: #dddddd;
                padding: 6px; border: 1px solid #333355;
                font-family: 'Courier New', monospace;
            }
            QComboBox QAbstractItemView { background: #1e1e2e; color: #ddd; }
        """)

        # ── estado de datos ──
        self.time_data     = []
        self.thrust_n      = []
        self.impulse_data  = []
        self.rssi_data     = []
        self.lost_pkts     = 0
        self.last_seq      = -1
        self.start_ts      = None
        self.is_kg         = False
        self.tare_offset_n = 0.0
        self.active_samples_count = 0
        self.firing        = False

        # ── modo de conexión ──
        self.conn_mode = MODE_CABLE   # valor inicial

        self._build_ui()

    # ══════════════════════════════════════════
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
            self.lbl_logo.setPixmap(
                pix.scaledToHeight(52, Qt.TransformationMode.SmoothTransformation))
        else:
            self.lbl_logo.setText("HORUS")
            self.lbl_logo.setStyleSheet(
                "color:#00e5ff; font-size:22px; font-weight:bold;")

        lbl_sys = QLabel("SISTEMA DE TELEMETRÍA  //  BANCO DE PRUEBAS")
        lbl_sys.setStyleSheet("color:#5566aa; font-size:11px; letter-spacing:3px;")

        self.lbl_status = QLabel("● DESCONECTADO")
        self.lbl_status.setStyleSheet(
            "color:#ff4444; font-size:11px; font-weight:bold;")
        self.lbl_status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        hdr.addWidget(self.lbl_logo)
        hdr.addSpacing(16)
        hdr.addWidget(lbl_sys)
        hdr.addStretch()
        hdr.addWidget(self.lbl_status)
        vlay.addLayout(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #222244;")
        vlay.addWidget(sep)

        # ── Fila selector de modo ──
        mode_row = QHBoxLayout()

        lbl_mode = QLabel("MODO DE CONEXIÓN:")
        lbl_mode.setStyleSheet("color:#5566aa; font-size:10px; letter-spacing:2px;")

        self.btn_mode_cable = QPushButton("🔌  CABLE  (RS-485)")
        self.btn_mode_lora  = QPushButton("📡  LoRa")

        self.btn_mode_cable.clicked.connect(lambda: self._set_mode(MODE_CABLE))
        self.btn_mode_lora.clicked.connect( lambda: self._set_mode(MODE_LORA))

        mode_row.addWidget(lbl_mode)
        mode_row.addWidget(self.btn_mode_cable)
        mode_row.addWidget(self.btn_mode_lora)
        mode_row.addStretch()

        # Badge que muestra el modo activo
        self.lbl_mode_badge = QLabel()
        self.lbl_mode_badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        mode_row.addWidget(self.lbl_mode_badge)

        vlay.addLayout(mode_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #1a1a33;")
        vlay.addWidget(sep2)

        # ── Fila de controles ──
        ctrl = QHBoxLayout()
        self.port_combo = QComboBox()
        self._refresh_ports()

        self.btn_refresh  = QPushButton("↺ PUERTOS")
        self.btn_refresh.clicked.connect(self._refresh_ports)

        self.btn_connect  = QPushButton("⏚ CONECTAR")
        self.btn_connect.clicked.connect(self._toggle_connection)

        self.btn_unit = QPushButton("UNIDAD: N")
        self.btn_unit.setStyleSheet(
            "background:#1a1a3a; color:#00e5ff; border-color:#00e5ff44;")
        self.btn_unit.clicked.connect(self._toggle_unit)

        self.btn_tare = QPushButton("⊙  TARAR / CERO")
        self.btn_tare.setStyleSheet(
            "background:#1a1a2a; color:#ffcc00; border-color:#ffcc0044;")
        self.btn_tare.clicked.connect(self._tare)

        self.btn_clear = QPushButton("⌫ LIMPIAR")
        self.btn_clear.clicked.connect(self._clear_data)

        self.btn_export = QPushButton("↓ EXPORTAR CSV")
        self.btn_export.clicked.connect(self._export_csv)

        for w in [self.port_combo, self.btn_refresh, self.btn_connect,
                  self.btn_unit, self.btn_tare,
                  self.btn_clear, self.btn_export]:
            ctrl.addWidget(w)
        vlay.addLayout(ctrl)

        # ── KPI cards ──
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

        self.graph_thrust = pg.PlotWidget(title="")
        self._style_graph(self.graph_thrust,
                          "Tiempo (s)", "Empuje (N)", "#00e5ff")
        self.curve_thrust = self.graph_thrust.plot(
            pen=pg.mkPen(color='#00e5ff', width=2),
            fillLevel=0, brush=pg.mkBrush(color=(0, 229, 255, 25))
        )

        self.graph_impulse = pg.PlotWidget(title="")
        self._style_graph(self.graph_impulse,
                          "Tiempo (s)", "Impulso acum. (N·s)", "#00ff88")
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

        # Aplicar modo inicial
        self._set_mode(MODE_CABLE)

    # ══════════════════════════════════════════
    #  SELECTOR DE MODO
    # ══════════════════════════════════════════
    def _set_mode(self, mode):
        """Cambia el modo de conexión y actualiza la UI en consecuencia."""

        # No permitir cambio si hay conexión activa
        if hasattr(self, 'serial_thread') and self.serial_thread.isRunning():
            return

        self.conn_mode = mode

        CABLE_ACTIVE = (
            "background:#0d2b1a; color:#00ff88; "
            "border-color:#00ff8866; font-weight:bold;"
        )
        LORA_ACTIVE = (
            "background:#1a0d2b; color:#aa88ff; "
            "border-color:#aa88ff66; font-weight:bold;"
        )
        INACTIVE = ""

        if mode == MODE_CABLE:
            self.btn_mode_cable.setStyleSheet(CABLE_ACTIVE)
            self.btn_mode_lora.setStyleSheet(INACTIVE)

            self.lbl_mode_badge.setText("🔌  RS-485 / USB")
            self.lbl_mode_badge.setStyleSheet(
                "color:#00ff88; font-size:10px; letter-spacing:2px; "
                "background:#0d1a0d; border:1px solid #00ff8833; "
                "border-radius:4px; padding:4px 10px;")

            # RSSI oculto en modo cable
            self.kpi_rssi.setVisible(False)

        else:  # LoRa
            self.btn_mode_cable.setStyleSheet(INACTIVE)
            self.btn_mode_lora.setStyleSheet(LORA_ACTIVE)

            self.lbl_mode_badge.setText("📡  LoRa  (9-byte packet)")
            self.lbl_mode_badge.setStyleSheet(
                "color:#aa88ff; font-size:10px; letter-spacing:2px; "
                "background:#1a0d2b; border:1px solid #aa88ff33; "
                "border-radius:4px; padding:4px 10px;")

            # RSSI visible en modo LoRa
            self.kpi_rssi.setVisible(True)

    # ══════════════════════════════════════════
    def _style_graph(self, gw, xlabel, ylabel, color):
        gw.setBackground('#0d0d1a')
        gw.showGrid(x=True, y=True, alpha=0.15)
        gw.setLabel('left',   ylabel, color=color,    size='10pt')
        gw.setLabel('bottom', xlabel, color='#445566', size='9pt')
        gw.getPlotItem().titleLabel.setText(
            f"<span style='color:{color};font-size:11pt;letter-spacing:2px'>"
            f"{ylabel.upper()}</span>"
        )
        for ax in ['left', 'bottom']:
            gw.getAxis(ax).setPen(pg.mkPen('#222244'))
            gw.getAxis(ax).setTextPen(pg.mkPen('#556688'))

    def _refresh_ports(self):
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        ports += glob.glob('/dev/ttyV*')
        ports += glob.glob('/tmp/tty*')
        ports += glob.glob('/dev/pts/*')
        ports = sorted(set(ports))
        self.port_combo.addItems(ports if ports else ["(sin puertos)"])

    # ══════════════════════════════════════════
    def _toggle_connection(self):
        if hasattr(self, 'serial_thread') and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.btn_connect.setText("⏚ CONECTAR")
            self.btn_connect.setStyleSheet("")
            self.lbl_status.setText("● DESCONECTADO")
            self.lbl_status.setStyleSheet(
                "color:#ff4444; font-size:11px; font-weight:bold;")
            # Rehabilitar botones de modo al desconectar
            self.btn_mode_cable.setEnabled(True)
            self.btn_mode_lora.setEnabled(True)
        else:
            port = self.port_combo.currentText()
            if not port or port.startswith("("):
                return

            self.serial_thread = SerialReader(port, 115200, mode=self.conn_mode)
            self.serial_thread.data_received.connect(self._on_data)
            self.serial_thread.start()
            self.start_ts = time.time()

            color  = "#00ff88" if self.conn_mode == MODE_CABLE else "#aa88ff"
            label  = "RS-485" if self.conn_mode == MODE_CABLE else "LoRa"
            self.btn_connect.setText("⏚ DESCONECTAR")
            self.btn_connect.setStyleSheet(
                f"background:#1a2a1a; color:{color}; border-color:{color}44;")
            self.lbl_status.setText(f"● CONECTADO  [{label}]")
            self.lbl_status.setStyleSheet(
                f"color:{color}; font-size:11px; font-weight:bold;")

            # Bloquear cambio de modo mientras hay conexión activa
            self.btn_mode_cable.setEnabled(False)
            self.btn_mode_lora.setEnabled(False)

    # ══════════════════════════════════════════
    def _on_data(self, newtons, seq, rssi):
        if self.start_ts is None:
            self.start_ts = time.time()

        t = time.time() - self.start_ts

        # Paquetes perdidos
        if self.last_seq >= 0 and seq > self.last_seq + 1:
            self.lost_pkts += seq - self.last_seq - 1
        self.last_seq = seq

        # Aplicar tara y umbral de ruido
        newtons      = abs(newtons) - self.tare_offset_n
        thrust_clean = newtons if newtons >= THRUST_THRESHOLD else 0.0

        # Detección de disparo activo
        if thrust_clean > 0:
            self.active_samples_count += 1
            if self.active_samples_count >= ACTIVE_SAMPLES:
                self.firing = True
        else:
            self.active_samples_count = 0
            if self.firing:
                self.firing = False

        self.time_data.append(t)
        self.thrust_n.append(thrust_clean)

        # Integración solo durante disparo activo
        if len(self.thrust_n) >= 2 and self.firing:
            dt   = self.time_data[-1] - self.time_data[-2]
            prev = self.impulse_data[-1] if self.impulse_data else 0.0
            self.impulse_data.append(
                prev + 0.5 * (self.thrust_n[-1] + self.thrust_n[-2]) * dt)
        else:
            prev = self.impulse_data[-1] if self.impulse_data else 0.0
            self.impulse_data.append(prev)

        if rssi != -999:
            self.rssi_data.append(rssi)

        self._refresh_display(t, rssi)

    # ══════════════════════════════════════════
    def _refresh_display(self, t_now, rssi):
        if not self.thrust_n:
            return

        factor = 1.0 / GRAVITY if self.is_kg else 1.0
        unit   = "kg"   if self.is_kg else "N"
        i_unit = "kg·s" if self.is_kg else "N·s"

        thrust_now = self.thrust_n[-1]  * factor
        max_thrust = max(self.thrust_n) * factor
        impulso    = self.impulse_data[-1]
        clase, pct = clasificar_motor(impulso)

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

        # RSSI — solo actualizar si estamos en modo LoRa
        if self.conn_mode == MODE_LORA:
            if rssi != -999:
                self.kpi_rssi.set_value(float(rssi), 0)
                # Color según calidad de señal
                if rssi >= -70:
                    color = "#00ff88"   # buena
                elif rssi >= -100:
                    color = "#ffcc00"   # media
                else:
                    color = "#ff4444"   # débil
                self.kpi_rssi.lbl_value.setStyleSheet(
                    f"color: {color}; font-size: 26px; font-weight: bold;")
            elif self.rssi_data:
                self.kpi_rssi.set_value(float(self.rssi_data[-1]), 0)

        # Estado del disparo en el badge de estado
        if self.firing:
            self.lbl_status.setText("🔥 DISPARO ACTIVO")
            self.lbl_status.setStyleSheet(
                "color:#ff4400; font-size:13px; font-weight:bold;")
        elif hasattr(self, 'serial_thread') and self.serial_thread.isRunning():
            label = "RS-485" if self.conn_mode == MODE_CABLE else "LoRa"
            color = "#00ff88" if self.conn_mode == MODE_CABLE else "#aa88ff"
            self.lbl_status.setText(f"● EN ESPERA  [{label}]")
            self.lbl_status.setStyleSheet(
                f"color:{color}; font-size:11px; font-weight:bold;")

        t_arr = np.array(self.time_data)
        self.curve_thrust.setData(t_arr,  np.array(self.thrust_n)     * factor)
        self.curve_impulse.setData(t_arr, np.array(self.impulse_data) * factor)

        self.lbl_packets.setText(f"Paquetes recibidos: {len(self.thrust_n)}")
        self.lbl_time.setText(f"t = {t_now:.3f} s")

    # ══════════════════════════════════════════
    def _tare(self):
        """Calcula offset con las últimas muestras y resetea el impulso."""
        if len(self.thrust_n) < 5:
            return
        samples = (self.thrust_n[-TARE_SAMPLES:]
                   if len(self.thrust_n) >= TARE_SAMPLES
                   else self.thrust_n[:])
        self.tare_offset_n = float(np.mean(samples))

        # Resetear solo integración y estado de disparo
        self.impulse_data         = [0.0] * len(self.impulse_data)
        self.lost_pkts            = 0
        self.last_seq             = -1
        self.active_samples_count = 0
        self.firing               = False

        self.kpi_impulse.lbl_value.setText("0.00")
        self.kpi_class.lbl_value.setText("—")

    def _toggle_unit(self):
        self.is_kg = not self.is_kg
        lbl = "kg" if self.is_kg else "N"
        self.btn_unit.setText(f"UNIDAD: {lbl}")
        self.graph_thrust.setLabel('left',  f"Empuje ({lbl})")
        self.graph_impulse.setLabel('left',
            f"Impulso acum. ({'kg·s' if self.is_kg else 'N·s'})")
        if self.thrust_n:
            self._refresh_display(self.time_data[-1], -999)

    def _clear_data(self):
        self.time_data.clear()
        self.thrust_n.clear()
        self.impulse_data.clear()
        self.rssi_data.clear()
        self.lost_pkts            = 0
        self.last_seq             = -1
        self.start_ts             = None
        self.tare_offset_n        = 0.0
        self.active_samples_count = 0
        self.firing               = False
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
            headers = ["Tiempo (s)", "Empuje (N)", "Empuje (kg)",
                       "Impulso acum (N·s)", "Impulso acum (kg·s)",
                       "Pkts perdidos"]
            if self.conn_mode == MODE_LORA:
                headers.insert(-1, "RSSI (dBm)")
            w.writerow(headers)

            for i in range(len(self.time_data)):
                row = [
                    round(self.time_data[i],    4),
                    round(self.thrust_n[i],     4),
                    round(self.thrust_n[i]    / GRAVITY, 4),
                    round(self.impulse_data[i], 4),
                    round(self.impulse_data[i] / GRAVITY, 4),
                ]
                if self.conn_mode == MODE_LORA:
                    row.append(self.rssi_data[i] if i < len(self.rssi_data) else "")
                row.append(self.lost_pkts)
                w.writerow(row)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Courier New", 9))
    win = RocketDashboard()
    win.show()
    sys.exit(app.exec())