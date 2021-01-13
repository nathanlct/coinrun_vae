import http.client, urllib

APP_TOKEN = ''
USER_TOKEN = ''


def notify(message, title='Training notification'):
    """Send a Pushover notification."""
    conn = http.client.HTTPSConnection('api.pushover.net:443')
    conn.request('POST', '/1/messages.json',
    urllib.parse.urlencode({
        'token': APP_TOKEN,
        'user': USER_TOKEN,
        'title': title,
        'message': message,
    }), { 'Content-type': 'application/x-www-form-urlencoded' })
