# in your server monitoring app
import requests
import socket

ALERT_ENDPOINT = "http://127.0.0.1:5015/push-alert"
MONITOR_SOURCE = socket.gethostname()  # or hardcode e.g. "server_monitor_dashboard"
TIMEOUT = 3  # seconds

def send_alert_to_socket(category, alert_type, message, server_name, metadata=None):
    payload = {
        "category": category,            # e.g. "server"
        "type": alert_type,              # e.g. "CPU", "Memory", "Offline"
        "message": message,
        "source": MONITOR_SOURCE,        # identifies THIS monitoring system
        "server": server_name,
        "metadata": metadata or {},
    }
    try:
        r = requests.post(ALERT_ENDPOINT, json=payload, timeout=TIMEOUT)
        if not r.ok:
            print("Failed to push alert:", r.status_code, r.text)
    except Exception as exc:
        print("Error pushing alert:", exc)






from flask import Blueprint, jsonify
# import the helper from above
from .alert_push import send_alert_to_socket   # or place helper in same file

bp = Blueprint('server_monitor', __name__)

@bp.route('/api/alerts')
def alerts():
    cache = load_cache()
    if not cache:
        cache = update_server_cache()
    
    alerts = []
    alert_servers = set()
    
    for server in cache['servers']:
        server_has_alert = False

        # 1) Server offline
        if server['status'] != 'online':
            msg = f"Server '{server['name']}' is offline"
            alert = {
                'server': server['name'],
                'type': 'Offline',
                'message': msg,
            }
            alerts.append(alert)
            alert_servers.add(server['name'])

            # push to central socket server
            send_alert_to_socket(
                category="server",
                alert_type="offline",
                message=msg,
                server_name=server['name'],
                metadata={"source_system": "server_monitor", "status": server['status']},
            )
            continue
        
        # 2) CPU high
        if server.get('cpu') and server['cpu'] > 75:
            msg = f"CPU usage is high ({server['cpu']}%)"
            alert = {
                'server': server['name'],
                'type': 'CPU',
                'message': msg,
            }
            alerts.append(alert)
            server_has_alert = True

            send_alert_to_socket(
                category="server",
                alert_type="cpu_high",
                message=msg,
                server_name=server['name'],
                metadata={"cpu": server['cpu']},
            )
        
        # 3) Memory high
        mem = server.get('memory')
        if mem and mem.get('usage_percent', 0) > 75:
            msg = f"Memory usage is high ({mem['usage_percent']}%)"
            alert = {
                'server': server['name'],
                'type': 'Memory',
                'message': msg,
            }
            alerts.append(alert)
            server_has_alert = True

            send_alert_to_socket(
                category="server",
                alert_type="memory_high",
                message=msg,
                server_name=server['name'],
                metadata={"memory_usage": mem['usage_percent']},
            )
        
        # 4) Storage high
        for storage in server.get('storage', []):
            if storage['mountpoint'] in ('/', '/root') and storage.get('percent', 0) > 75:
                msg = f"Root storage usage is high ({storage['percent']}%)"
                alert = {
                    'server': server['name'],
                    'type': 'Storage',
                    'message': msg,
                }
                alerts.append(alert)
                server_has_alert = True

                send_alert_to_socket(
                    category="server",
                    alert_type="storage_high",
                    message=msg,
                    server_name=server['name'],
                    metadata={
                        "mountpoint": storage['mountpoint'],
                        "percent": storage['percent'],
                    },
                )
                break
        
        if server_has_alert:
            alert_servers.add(server['name'])
    
    return jsonify({
        'alerts': alerts,
        'total_servers': len(cache['servers']),
        'alert_servers': len(alert_servers),
    })


