"""
Generate Infrastructure Health Monitoring - Product Pipeline Diagram
Shows the full data flow from vehicle sensors to maintenance notification,
incorporating Ericsson products at each stage.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_pipeline_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    # Title
    ax.text(10, 13.5, "Infrastructure Health Monitoring - Product Pipeline",
            ha="center", va="center", fontsize=18, fontweight="bold", color="#2c3e50")
    ax.text(10, 13.0, "From Vehicle Sensor Data to Maintenance Action",
            ha="center", va="center", fontsize=12, color="#7f8c8d")

    # --- Define pipeline stages ---
    stages = [
        {
            "x": 2.0, "y": 10.5, "w": 3.2, "h": 2.0,
            "title": "1. DATA GENERATION",
            "desc": "Vehicle sensors collect\nGPS + Accelerometer data\nin real-time",
            "detail": "Smartphones / OBD-II\nPhyphox app / Fleet IoT",
            "color": "#3498db",
        },
        {
            "x": 6.5, "y": 10.5, "w": 3.2, "h": 2.0,
            "title": "2. CONNECTIVITY",
            "desc": "Data transmitted via\ncellular network",
            "detail": "Ericsson Radio System\nEricsson 5G RAN",
            "color": "#2980b9",
            "ericsson": True,
        },
        {
            "x": 11.0, "y": 10.5, "w": 3.2, "h": 2.0,
            "title": "3. NETWORK CORE",
            "desc": "Data routed through\npacket core to cloud",
            "detail": "Ericsson Packet Core\nEricsson Cloud Core",
            "color": "#1a5276",
            "ericsson": True,
        },
        {
            "x": 15.5, "y": 10.5, "w": 3.2, "h": 2.0,
            "title": "4. IoT PLATFORM",
            "desc": "Device management &\ndata ingestion",
            "detail": "Ericsson IoT Accelerator\nEricsson DCP",
            "color": "#6c3483",
            "ericsson": True,
        },
        {
            "x": 15.5, "y": 7.0, "w": 3.2, "h": 2.0,
            "title": "5. EDGE PROCESSING",
            "desc": "Real-time stream\nprocessing & filtering",
            "detail": "Ericsson Edge Gravity\nEdge Computing (MEC)",
            "color": "#1e8449",
            "ericsson": True,
        },
        {
            "x": 11.0, "y": 7.0, "w": 3.2, "h": 2.0,
            "title": "6. ANALYTICS ENGINE",
            "desc": "Adaptive path model\nAnomaly detection AI",
            "detail": "Ericsson Network\nIntelligence (ENI)",
            "color": "#d35400",
            "ericsson": True,
        },
        {
            "x": 6.5, "y": 7.0, "w": 3.2, "h": 2.0,
            "title": "7. DECISION ENGINE",
            "desc": "Severity scoring &\nmaintenance prioritization",
            "detail": "Ericsson Expert\nAnalytics (EEA)",
            "color": "#c0392b",
            "ericsson": True,
        },
        {
            "x": 2.0, "y": 7.0, "w": 3.2, "h": 2.0,
            "title": "8. NOTIFICATION",
            "desc": "Alert dispatched to\nmaintenance teams",
            "detail": "Ericsson ServiceOn\nAPI Gateway",
            "color": "#f39c12",
            "ericsson": True,
        },
    ]

    # Draw stages
    for stage in stages:
        x, y, w, h = stage["x"], stage["y"], stage["w"], stage["h"]
        color = stage["color"]
        is_ericsson = stage.get("ericsson", False)

        # Box
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.9,
        )
        ax.add_patch(box)

        # Ericsson badge
        if is_ericsson:
            badge = FancyBboxPatch(
                (x + w/2 - 0.7, y + h/2 - 0.3), 0.6, 0.25,
                boxstyle="round,pad=0.02",
                facecolor="white", edgecolor=color, linewidth=1, alpha=0.9,
            )
            ax.add_patch(badge)
            ax.text(x + w/2 - 0.4, y + h/2 - 0.17, "E///",
                    ha="center", va="center", fontsize=7, color=color, fontweight="bold")

        # Title
        ax.text(x, y + h/2 - 0.35, stage["title"],
                ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
        # Description
        ax.text(x, y - 0.05, stage["desc"],
                ha="center", va="center", fontsize=8, color="white", alpha=0.95)
        # Detail (product names)
        ax.text(x, y - h/2 + 0.35, stage["detail"],
                ha="center", va="center", fontsize=7.5, color="white",
                fontstyle="italic", alpha=0.85)

    # --- Draw arrows between stages ---
    arrow_style = "Simple,tail_width=2,head_width=8,head_length=5"

    # Top row: left to right (stages 1->2->3->4)
    for i in range(3):
        x_start = stages[i]["x"] + stages[i]["w"]/2
        x_end = stages[i+1]["x"] - stages[i+1]["w"]/2
        y_pos = stages[i]["y"]
        ax.annotate("", xy=(x_end - 0.05, y_pos), xytext=(x_start + 0.05, y_pos),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.5))

    # Right side: top to bottom (stage 4 -> 5)
    ax.annotate("", xy=(stages[4]["x"], stages[4]["y"] + stages[4]["h"]/2 + 0.05),
                xytext=(stages[3]["x"], stages[3]["y"] - stages[3]["h"]/2 - 0.05),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.5))

    # Bottom row: right to left (stages 5->6->7->8)
    for i in range(4, 7):
        x_start = stages[i]["x"] - stages[i]["w"]/2
        x_end = stages[i+1]["x"] + stages[i+1]["w"]/2
        y_pos = stages[i]["y"]
        ax.annotate("", xy=(x_end + 0.05, y_pos), xytext=(x_start - 0.05, y_pos),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.5))

    # --- Customer/End result at the bottom ---
    # Maintenance team box
    cx, cy, cw, ch = 10.0, 3.5, 6.0, 2.2
    customer_box = FancyBboxPatch(
        (cx - cw/2, cy - ch/2), cw, ch,
        boxstyle="round,pad=0.15",
        facecolor="#27ae60", edgecolor="#1e8449", linewidth=3, alpha=0.9,
    )
    ax.add_patch(customer_box)
    ax.text(cx, cy + 0.6, "MAINTENANCE TEAM / CITY AUTHORITY",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax.text(cx, cy, "Receives prioritized alert with:",
            ha="center", va="center", fontsize=9.5, color="white")
    ax.text(cx, cy - 0.55,
            "- GPS location of detected issue (pothole, road damage)\n"
            "- Severity score & confidence level\n"
            "- Historical trend data & estimated urgency",
            ha="center", va="center", fontsize=8.5, color="white", alpha=0.9)

    # Arrow from notification stage to customer
    ax.annotate("", xy=(cx - 1.5, cy + ch/2 + 0.05),
                xytext=(stages[7]["x"], stages[7]["y"] - stages[7]["h"]/2 - 0.05),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=3,
                               connectionstyle="arc3,rad=0.2"))

    # --- Legend / Key ---
    ax.text(1.0, 1.8, "Ericsson Products Used:", fontsize=10, fontweight="bold", color="#2c3e50")
    products = [
        ("Ericsson Radio System / 5G RAN", "Cellular connectivity for vehicle data upload"),
        ("Ericsson Packet Core / Cloud Core", "Reliable data routing from device to cloud"),
        ("Ericsson IoT Accelerator / DCP", "Device connectivity platform & SIM management"),
        ("Ericsson Edge Gravity (MEC)", "Low-latency edge processing for real-time filtering"),
        ("Ericsson Network Intelligence (ENI)", "AI/ML analytics for pattern detection"),
        ("Ericsson Expert Analytics (EEA)", "Decision support & event correlation"),
        ("Ericsson ServiceOn", "Service orchestration & alert dispatch"),
    ]
    for i, (name, desc) in enumerate(products):
        y_pos = 1.3 - i * 0.35
        ax.text(1.2, y_pos, f"E///  {name}", fontsize=7.5, fontweight="bold", color="#1a5276")
        ax.text(8.5, y_pos, desc, fontsize=7.5, color="#555")

    # Data flow label
    ax.text(10, 12.3, "--- Data Flow Direction --->",
            ha="center", va="center", fontsize=9, color="#95a5a6", fontstyle="italic")

    plt.tight_layout(pad=1.0)
    output_path = r"C:\Users\eyorbdt\STONK\infrastructure_health_network\pipeline_diagram.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    print(f"Pipeline diagram saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_pipeline_diagram()
