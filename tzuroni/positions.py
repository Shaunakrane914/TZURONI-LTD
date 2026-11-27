import re
import unicodedata

from .assets import ASSETS


SPEAKER = r"(?:i(?:\s+am|['’]m)|we(?:\s+are|['’]re))\s+(?:(?:still|currently)\s+)?"
CLAUSE_END = re.compile(r'[!?;。！？；\n]+|\.(?=\s|$)')
DISCLAIMED = re.compile(
    r"\b(?:not|never|no longer|closed|covered|sold|don['’]t|if)\b|不再|未有|已平倉|沒有持有|如果|假如|"
    r'보유하지|보유 안|청산했|포지션이 없|보유하고 있지|아닙니다'
)
LONG = re.compile(
    rf"\b{SPEAKER}long\b(?!\s+overdue)|\b(?:i|we)\s+(?:(?:currently|still)\s+)?"
    r'(?:hold|own)\s+(?:(?:a|the|some|my|our)\s+)?(?:long\s+position|shares)\b|'
    r'(?:我|本人)(?:目前|現在|而家|仍然)?(?:持有|揸住|持倉|持仓|持有多頭|持有多头)|'
    r'(?:저는|나는|제가|내가|저희는)\s*[^.!?]{0,30}?(?:주식(?:을)?\s*(?:보유\s*중|보유하고 있)|롱\s*포지션(?:을)?\s*보유)'
)
SHORT = re.compile(
    rf"\b{SPEAKER}short\b(?!\s+(?:of|on\s+cash))|\b(?:i|we)\s+(?:(?:currently|still)\s+)?hold\s+"
    r'(?:(?:a|the|my|our)\s+)?short\s+position\b|'
    r'(?:我|本人)(?:目前|現在|而家|仍然)?(?:持有[^,，。!?]{0,14}?(?:空頭|空头|空倉|空仓)|有空倉|有空仓|做緊空)|'
    r'(?:저는|나는|제가|내가|저희는)\s*[^.!?]{0,30}?숏\s*포지션(?:을)?\s*보유'
)


def disclosed_positions(text, asset):
    """Find explicit current positions disclosed in a social post."""
    text = unicodedata.normalize('NFKC', text).casefold()
    positions = set()
    for clause in CLAUSE_END.split(text):
        if DISCLAIMED.search(clause):
            continue
        short = bool(SHORT.search(clause))
        long = bool(LONG.search(clause))
        if not (long or short):
            continue
        mentions = [candidate.ticker for candidate in ASSETS if candidate.mentioned_in(clause)]
        if not mentions:
            mentions = [candidate.ticker for candidate in ASSETS if candidate.mentioned_in(text)]
        if mentions and mentions != [asset.ticker]:
            continue
        if short:
            positions.add('short')
        elif long:
            positions.add('long')
    return positions
