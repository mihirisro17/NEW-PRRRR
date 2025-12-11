@bp.route('/api/alerts')
def alerts():
    cache = load_cache()
    if not cache:
        cache = update_server_cache()
    
    alerts = []
    alert_servers = set()
    
    for server in cache['servers']:
        server_has_alert = False
        if server['status'] != 'online':
            alerts.append({
                'server': server['name'],
                'type': 'Offline',
                'message': f"Server '{server['name']}' is offline"
            })
            alert_servers.add(server['name'])
            continue
        
        if server.get('cpu') and server['cpu'] > 75:
            alerts.append({
                'server': server['name'],
                'type': 'CPU',
                'message': f"CPU usage is high ({server['cpu']}%)"
            })
            server_has_alert = True
        
        mem = server.get('memory')
        if mem and mem.get('usage_percent', 0) > 75:
            alerts.append({
                'server': server['name'],
                'type': 'Memory',
                'message': f"Memory usage is high ({mem['usage_percent']}%)"
            })
            server_has_alert = True
        
        for storage in server.get('storage', []):
            if storage['mountpoint'] in ('/', '/root') and storage.get('percent', 0) > 75:
                alerts.append({
                    'server': server['name'],
                    'type': 'Storage',
                    'message': f"Root storage usage is high ({storage['percent']}%)"
                })
                server_has_alert = True
                break
        
        if server_has_alert:
            alert_servers.add(server['name'])
    
    return jsonify({
        'alerts': alerts,
        'total_servers': len(cache['servers']),
        'alert_servers': len(alert_servers)
    })
