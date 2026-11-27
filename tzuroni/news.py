import re

import requests
from exa_py import Exa


def create_news_client():
    return Exa(base_url='https://api.exa.ai', timeout=35)


def search_news(client, asset, start, end):
    local_name = asset.aliases[0]
    query = f'Recent financial news on {asset.name} ({asset.ticker}, {local_name})'
    objective = (
        'Build a time-aligned stock sentiment feature. Prioritize timely financial reporting, '
        'company disclosures, and exchange announcements about this listed company. Exclude '
        'generic market recaps with no material company update and duplicate syndications. '
        'Pull facts about earnings, guidance, orders, product developments, or analyst revisions.'
    )
    try:
        response = client.search(
            query,
            objective=objective,
            start_published_date=start.isoformat(),
            end_published_date=end.isoformat(),
            contents={'highlights': True},
        )
    except (requests.RequestException, ValueError) as error:
        # SDK errors can include a provider response body. Report only status or error type.
        status = re.match(r'Request failed with status code (\d{3}):', str(error))
        detail = f'HTTP {status[1]}' if status else type(error).__name__
        raise RuntimeError(f'Exa {asset.ticker}: {detail}') from None

    return [
        dict(id=item.id, url=item.url, title=item.title, publishedDate=item.published_date,
             highlights=item.highlights or [])
        for item in response.results
    ]
