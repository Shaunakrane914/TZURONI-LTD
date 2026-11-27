from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    exchange: str
    currency: str
    aliases: tuple[str, ...]

    @property
    def code(self) -> str:
        return self.ticker.split(".")[0]

    @property
    def search_terms(self) -> tuple[str, ...]:
        return (self.ticker, f"${self.code}", self.name, *self.aliases)

    def mentioned_in(self, text: str) -> bool:
        text = unicodedata.normalize("NFKC", text).casefold()
        for term in self.search_terms:
            term = term.casefold()
            pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
            if term.isascii() and re.search(pattern, text):
                return True
            if not term.isascii() and term in text:
                return True

        if self.exchange == "HKEX":
            code = re.escape(self.code.lstrip("0"))
            return bool(re.search(rf"(?<!\d)(?:\$0*{code}\b|0*{code}[. -]?hk\b)", text))
        return bool(re.search(rf"(?<![a-z0-9]){self.code}(?![a-z0-9])", text))


ASSETS = (
    Asset("0700.HK", "Tencent", "HKEX", "HKD", ("騰訊", "腾讯")),
    Asset("9988.HK", "Alibaba", "HKEX", "HKD", ("阿里巴巴", "BABA")),
    Asset("3690.HK", "Meituan", "HKEX", "HKD", ("美團", "美团")),
    Asset("1810.HK", "Xiaomi", "HKEX", "HKD", ("小米",)),
    Asset("1211.HK", "BYD", "HKEX", "HKD", ("比亞迪", "比亚迪")),
    Asset("005930.KS", "Samsung Electronics", "KRX", "KRW", ("삼성전자",)),
    Asset("000660.KS", "SK Hynix", "KRX", "KRW", ("SK하이닉스", "하이닉스")),
    Asset("005380.KS", "Hyundai Motor", "KRX", "KRW", ("현대차", "현대자동차")),
    Asset("035420.KS", "NAVER", "KRX", "KRW", ("네이버",)),
    Asset("035720.KS", "Kakao", "KRX", "KRW", ("카카오",)),
)

SOURCES = {"HKEX": ("lihkg", "telegram"), "KRX": ("naver", "telegram")}
