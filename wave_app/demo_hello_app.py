"""
Wave App doing 'Hello World'
"""

from h2o_wave import Q, main, app, ui

@app('/hello_app')
async def serve(q: Q):

    q.page['card_hello'] = ui.markdown_card(
        box = '1 1 2 3',
        title = 'Hello World!',
        content = '初めてのH2O Wave (wave app)',
    )

    await q.page.save()
