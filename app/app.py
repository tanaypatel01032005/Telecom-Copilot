import streamlit as st
import time
from datetime import datetime
import plotly.graph_objects as go
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Telecom AI Copilot NOC",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PIPELINE INITIALIZATION (REAL BACKEND)
# ==========================================
def load_pipeline():
    try:
        from src.pipeline.inference_pipeline import TelecomCopilot
        return TelecomCopilot()
    except Exception as e:
        # Fallback for when models are not yet trained
        print(f"DEBUG: Pipeline load failed: {e}")
        return None

pipeline = load_pipeline()
# Removed blocking warmup call
# ==========================================
# CUSTOM CSS (Refined Cyberpunk / Glassmorphism)
# ==========================================
def inject_custom_css():
    custom_css = """
    <style>
        /* Global App Background */
        .stApp {
            background-color: #060913;
            color: #e0e6ed;
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide top header bar */
        header {visibility: hidden;}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0a0f1d !important;
            border-right: 1px solid rgba(0, 243, 255, 0.1);
        }
        [data-testid="stChatMessage"] {
            background: rgba(14, 20, 35, 0.75);
            border: 1px solid rgba(0,243,255,0.08);
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 12px;
        }
        /* Neon Glow Text */
        .glow-text {
            color: #00f3ff;
            text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
            font-family: 'Courier New', Courier, monospace;
            font-weight: bold;
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(145deg, rgba(16, 24, 39, 0.8) 0%, rgba(10, 15, 26, 0.9) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 243, 255, 0.2);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 4px 16px 0 rgba(0, 243, 255, 0.05);
            margin-bottom: 15px;
        }
        .metric-title { color: #8b9bb4; font-size: 0.85rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;}
        .metric-value { color: #ffffff; font-size: 1.6rem; font-weight: bold; margin-top: 4px; }
        
        /* Alert Items */
        .alert-item {
            background: rgba(16, 24, 39, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-left: 3px solid #00f3ff;
            padding: 10px 14px;
            border-radius: 4px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
        }
        .alert-item.critical { border-left-color: #ff3366; }
        .alert-item.warning { border-left-color: #ffbb00; }
        .badge { padding: 3px 6px; border-radius: 3px; font-size: 0.7rem; font-weight: bold; text-transform: uppercase;}
        .badge.critical { background: rgba(255, 51, 102, 0.15); color: #ff3366; }
        .badge.warning { background: rgba(255, 187, 0, 0.15); color: #ffbb00; }
        .badge.healthy { background: rgba(0, 255, 136, 0.15); color: #00ff88; }

        /* Ticket Popup UI */
        .ticket-banner {
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid #00ff88;
            border-radius: 6px;
            padding: 10px;
            margin-top: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .ticket-icon { font-size: 1.5rem; }
        .ticket-text { color: #00ff88; font-family: monospace; font-size: 0.9rem;}
        
        /* Citation Pills */
        .citation-pill {
            display: inline-block;
            background: rgba(0, 243, 255, 0.1);
            border: 1px solid rgba(0, 243, 255, 0.3);
            color: #00f3ff;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            margin-right: 5px;
            margin-top: 5px;
            font-family: monospace;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Operations center AI initialized. Real-time RAG pipeline connected. How can I assist you today?", "metadata": None}
        ]
    if "show_copilot" not in st.session_state:
        st.session_state.show_copilot = False

# ==========================================
# UI COMPONENTS (MODULAR)
# ==========================================
def render_metric_card(title, value, delta, color="#00ff88"):
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value} <span style="color: {color}; font-size: 0.8rem; margin-left: 8px;">{delta}</span></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_alert(text, severity="critical"):
    html = f"""
    <div class="alert-item {severity}">
        <span>{text}</span>
        <span class="badge {severity}">{severity}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def generate_network_map():
    # Retaining existing Plotly map component
    cities = {
        'Mumbai': (19.0760, 72.8777), 'Delhi': (28.7041, 77.1025),
        'London': (51.5074, -0.1278), 'New York': (40.7128, -74.0060),
        'Tokyo': (35.6762, 139.6503), 'Singapore': (1.3521, 103.8198),
        'Sydney': (-33.8688, 151.2093), 'Frankfurt': (50.1109, 8.6821)
    }
    fig = go.Figure()
    
    lats = [c[0] for c in cities.values()]
    lons = [c[1] for c in cities.values()]
    names = list(cities.keys())
    
    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats, text=names, mode='markers+text', textposition="bottom center",
        textfont=dict(color='#8b9bb4', size=10),
        marker=dict(size=6, color='#00f3ff', opacity=0.8)
    ))
    
    routes = [('Mumbai', 'Singapore'), ('Singapore', 'Tokyo'), ('Tokyo', 'Sydney'),
              ('Mumbai', 'Frankfurt'), ('Frankfurt', 'London'), ('London', 'New York'),
              ('New York', 'Frankfurt'), ('Delhi', 'Mumbai')]
    
    for route in routes:
        fig.add_trace(go.Scattergeo(
            lon=[cities[route[0]][1], cities[route[1]][1]],
            lat=[cities[route[0]][0], cities[route[1]][0]],
            mode='lines', line=dict(width=1.5, color='#00f3ff'), opacity=0.3
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            bgcolor='rgba(0,0,0,0)', showland=True, landcolor='#0d1421',
            showocean=True, oceancolor='#060913', showcountries=True, countrycolor='#1a2436',
            coastlinecolor='#1a2436', projection_type='equirectangular'
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, height=350
    )
    return fig

def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 class='glow-text'>🌐 TELECOM<br>&nbsp;&nbsp;COPILOT</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.radio("Navigation", ["Dashboard", "Network Map", "Alerts", "Analytics", "Security"], label_visibility="collapsed")
        st.markdown("---")
        
        button_label = "✖ Close AI Copilot" if st.session_state.show_copilot else "💬 Open AI Copilot"
        if st.button(button_label, use_container_width=True):
            st.session_state.show_copilot = not st.session_state.show_copilot
            st.rerun()

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='color: #8b9bb4; font-size: 0.8rem;'>SYSTEM STATUS</div>", unsafe_allow_html=True)
        st.markdown("<div style='color: #00ff88; font-size: 0.9rem;'>● Backend: Connected</div>", unsafe_allow_html=True)
        st.markdown("<div style='color: #00ff88; font-size: 0.9rem;'>● FAISS Index: Loaded</div>", unsafe_allow_html=True)

def render_main_dashboard():
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1: render_metric_card("Active Connections", "1.2M", "▲ 12.8%", "#00ff88")
    with m2: render_metric_card("Global Throughput", "84.5 Tbps", "▲ 2.4%", "#00ff88")
    with m3: render_metric_card("Avg Latency", "46.8 ms", "▼ 1.2ms", "#00ff88")
    with m4: render_metric_card("Escalation Rate", "4.2%", "▼ 0.5%", "#00ff88")

    # Network Visualization
    st.markdown("<h4 style='color: #e0e6ed; font-size: 1.1rem;'>Live Network Telemetry</h4>", unsafe_allow_html=True)
    st.plotly_chart(generate_network_map(), use_container_width=True, config={'displayModeBar': False})

    # Outages & Security
    lower_col1, lower_col2 = st.columns(2)
    with lower_col1:
        st.markdown("<h5 style='color: #e0e6ed; font-size: 1rem;'>Active Outages</h5>", unsafe_allow_html=True)
        render_alert("Mumbai Core - Partial Latency Spikes", "warning")
        render_alert("Dublin Node - Resolving Fiber Cut", "critical")
        
    with lower_col2:
        st.markdown("<h5 style='color: #e0e6ed; font-size: 1rem;'>Security Telemetry</h5>", unsafe_allow_html=True)
        render_alert("DDoS Attempt - Frankfurt Gateway", "critical")
        render_alert("BGP Route Flapping - Mitigated", "healthy")

def render_copilot_panel():
    st.markdown("""
    <div style='border-bottom: 1px solid rgba(0,243,255,0.2); padding-bottom: 10px; margin-bottom: 15px;'>
        <h4 style='color: #00f3ff; margin: 0; font-size: 1.1rem;'>AI Copilot Assistant</h4>
    </div>
    """, unsafe_allow_html=True)
    
    chat_container = st.container(height=550, border=False)
    
    with chat_container:
        # Render historical messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])
                
                # Render Metadata if it exists
                if msg.get("metadata"):
                    meta = msg["metadata"]
                    
                    # Citations
                    if meta.get("citations"):
                       # citations_html = "".join([f"<span class='citation-pill'>📄 {c}</span>" for c in meta["citations"]])
                        citations_html = "".join([
                            f"<span class='citation-pill'>📄 {c.get('doc_id', 'unknown')}</span>"
                            if isinstance(c, dict)
                            else f"<span class='citation-pill'>📄 {c}</span>"
                            for c in meta["citations"]
                        ])

                        st.markdown(citations_html, unsafe_allow_html=True)
                    
                    # Escalation Ticket Popup
                    if meta.get("escalated") and meta.get("ticket_id"):
                        ticket_html = f"""
                        <div class="ticket-banner">
                            <span class="ticket-icon">🎟️</span>
                            <span class="ticket-text">Ticket Created: <b>{meta['ticket_id']}</b><br>Escalated to L2 Operations</span>
                        </div>
                        """
                        st.markdown(ticket_html, unsafe_allow_html=True)
                    
                    # Tool Execution Debug Logs
                    with st.expander(f"⚙️ Pipeline Execution Log ({meta.get('latency_ms', 0):.0f}ms)"):
                        st.json({
                            "tools_used": meta.get("tools_used", []),
                            "citations": meta.get("citations", []),
                            "escalated": meta.get("escalated", False),
                            "ticket_id": meta.get("ticket_id", None)
                        })

    # Chat Input
    if user_input := st.chat_input("Query network status, SOPs, or escalate..."):
        st.session_state.messages.append({"role": "user", "content": user_input, "metadata": None})
        st.rerun()

def process_ai_response():
    if not st.session_state.messages or st.session_state.messages[-1]["role"] != "user":
        return
        
    user_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant", avatar="🤖"):
        if pipeline is None:
            st.error("Backend pipeline not initialized.")
            return

        # UI container for live tool execution status
        status_container = st.status("Initializing AI Pipeline...", expanded=True)
        
        try:
            # status_container.update(label="Running Retriever & Tools...", state="running")
            # CALLING THE REAL BACKEND PIPELINE
            def update_ui_status(msg):
                status_container.write(f"⏳ {msg}")
            
            result = pipeline.run(user_query, status_callback=update_ui_status)
            
            status_container.update(label="Response Generated", state="complete", expanded=False)

            # Simulated Streaming for UX (Reads the complete string and yields chunks)
            def stream_response(text):
                for word in text.split(" "):
                    yield word + " "
                    time.sleep(0.02)
            
            # st.write_stream(stream_response(result.get("answer", ""))) 
            response_text = result.get("answer") or result.get("response") or "No response generated."

            # Clean ugly backend formatting
            response_text = response_text.replace("[Doc", "")
            response_text = response_text.replace("]", "")
            response_text = response_text.replace("doc_id", "")
            response_text = response_text.replace("section_id", "")
            response_text = response_text.strip()
            if len(response_text) < 3:
             response_text = "I could not generate a confident response for this query."
            st.write_stream(stream_response(response_text))
            
            # Handle Metadata rendering immediately
            metadata = {
                "citations": result.get("citations", []),
                # "tools_used": result.get("tools_used", []),
                "tools_used": [t["tool"] for t in result.get("tool_trace", [])],
                "escalated": result.get("escalated", False),
                "ticket_id": result.get("ticket_id", None),
                "latency_ms": result.get("latency_ms", 0.0)
            }
            
            if metadata["citations"]:
                # st.markdown("".join([f"<span class='citation-pill'>📄 {c}</span>" for c in metadata["citations"]]), unsafe_allow_html=True)
                citation_html = "".join([
                f"<span class='citation-pill'>📄 {c.get('doc_id', 'unknown')}</span>"
                if isinstance(c, dict)
                else f"<span class='citation-pill'>📄 {c}</span>"
                for c in metadata["citations"]
            ])
                st.markdown(citation_html, unsafe_allow_html=True)
                
            if metadata["escalated"] and metadata["ticket_id"]:
                st.markdown(f"""
                <div class="ticket-banner">
                    <span class="ticket-icon">🎟️</span>
                    <span class="ticket-text">Ticket Created: <b>{metadata['ticket_id']}</b><br>Escalated to L2 Operations</span>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"⚙️ AI Pipeline Trace • {metadata['latency_ms']:.0f} ms"):

                    st.markdown("### Tools Used")
                    if metadata["tools_used"]:
                        for tool in metadata["tools_used"]:
                            st.success(f"✓ {tool}")
                    else:
                        st.info("No external tools executed.")

                    st.markdown("### Escalation")
                    st.write("Yes" if metadata["escalated"] else "No")

                    if metadata["ticket_id"]:
                        st.write(f"Ticket ID: {metadata['ticket_id']}")

                    st.markdown("### Citations")
                    if metadata["citations"]:
                        st.json(metadata["citations"])
                    else:
                        st.write("No citations returned.")
            # Save to session state
            st.session_state.messages.append({
                "role": "assistant",
                # "content": result.get("answer", ""),
                "content": response_text,
                "metadata": metadata
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            status_container.update(label="Pipeline Error", state="error", expanded=True)
            # st.error(f"Error executing pipeline: {str(e)}")
            st.error("AI Copilot encountered an internal processing issue.")

            with st.expander("Technical Logs"):
                st.code(str(e))

# ==========================================
# MAIN APPLICATION LOOP
# ==========================================
def main():
    inject_custom_css()
    init_session_state()
    render_sidebar()
    
    # Top Header
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
        <h3 style='margin:0;'>NOC Intelligence Dashboard</h3>
        <span style='color: #8b9bb4; font-size: 0.9rem;'>🔔 Admin User</span>
    </div>
    """, unsafe_allow_html=True)

    # Layout Handling based on Copilot State
    if st.session_state.show_copilot:
        col_main, col_chat = st.columns([7, 4], gap="large")
        with col_main:
            render_main_dashboard()
        with col_chat:
            render_copilot_panel()
            process_ai_response()
    else:
        col_main = st.container()
        with col_main:
            render_main_dashboard()

if __name__ == "__main__":
    
    main()