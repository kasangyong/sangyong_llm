"""검색 제공자 클라이언트.

제공자마다 요청 방식도 응답 필드도 다르다. Provider가 그 차이를 흡수하고
바깥에는 SearchResult 하나로만 내보낸다.

  Brave  GET  https://api.search.brave.com/res/v1/web/search?q=..&count=..
         헤더 X-Subscription-Token / 결과 web.results[] {title, url, description}
  Tavily POST https://api.tavily.com/search
         헤더 Authorization: Bearer / 본문 {query, max_results}
         결과 results[] {title, url, content, score}
  Serper POST https://google.serper.dev/search
         헤더 X-API-KEY / 본문 {q, num} / 결과 organic[] {title, link, snippet}

HTTP는 주입 가능한 fetch 함수로 뺐다. 테스트에서 진짜 네트워크를 치지 않기
위해서다. 기본 구현은 표준 라이브러리 urllib만 쓴다(새 패키지 설치 금지).

실패는 전부 예외다. 키가 없거나 429가 오는데 빈 리스트를 돌려주면
"검색 결과가 없었다"와 구분이 안 되고, 그 상태로 모델이 답을 지어낸다.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

DEFAULT_TIMEOUT = 10.0
DEFAULT_COUNT = 5


class SearchError(Exception):
    """검색 계층의 모든 실패의 상위 예외."""


class MissingAPIKeyError(SearchError):
    pass


class RateLimitError(SearchError):
    pass


class AuthError(SearchError):
    pass


class SearchTimeout(SearchError):
    pass


class NetworkError(SearchError):
    pass


class ProviderError(SearchError):
    """제공자가 준 오류 응답(4xx/5xx)."""


class MalformedResponseError(SearchError):
    """JSON이 깨졌거나 기대한 필드가 통째로 없다."""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    rank: int
    provider: str


@dataclass
class HTTPRequest:
    url: str
    method: str
    headers: dict
    body: bytes | None
    timeout: float


@dataclass
class HTTPResponse:
    status: int
    body: bytes


# ------------------------------------------------------------------- 제공자


@dataclass
class Provider:
    name: str = ""
    env_keys: tuple = ()

    def build(self, query: str, count: int, api_key: str) -> HTTPRequest:
        raise NotImplementedError

    def parse(self, payload: dict) -> list[SearchResult]:
        raise NotImplementedError

    def _collect(self, items, url_key: str, snippet_key: str) -> list[SearchResult]:
        """공통 정규화. url이 없는 항목은 쓸모가 없으니 버린다."""
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            url = (it.get(url_key) or "").strip()
            if not url:
                continue
            out.append(
                SearchResult(
                    title=(it.get("title") or "").strip(),
                    url=url,
                    snippet=(it.get(snippet_key) or "").strip(),
                    rank=len(out) + 1,
                    provider=self.name,
                )
            )
        if items and not out:
            raise MalformedResponseError(
                f"{self.name}: 결과 항목은 있는데 {url_key} 필드가 하나도 없다"
            )
        return out


class BraveProvider(Provider):
    def __init__(self):
        super().__init__("brave", ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"))

    def build(self, query: str, count: int, api_key: str) -> HTTPRequest:
        qs = urllib.parse.urlencode(
            {"q": query, "count": min(max(count, 1), 20)},
            quote_via=urllib.parse.quote,
        )
        return HTTPRequest(
            url=f"https://api.search.brave.com/res/v1/web/search?{qs}",
            method="GET",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            body=None,
            timeout=DEFAULT_TIMEOUT,
        )

    def parse(self, payload: dict) -> list[SearchResult]:
        web = payload.get("web")
        if not isinstance(web, dict) or not isinstance(web.get("results"), list):
            raise MalformedResponseError("brave: web.results가 없다")
        return self._collect(web["results"], "url", "description")


class TavilyProvider(Provider):
    def __init__(self):
        super().__init__("tavily", ("TAVILY_API_KEY",))

    def build(self, query: str, count: int, api_key: str) -> HTTPRequest:
        body = json.dumps(
            {
                "query": query,
                "max_results": min(max(count, 1), 20),
                # 요약(answer)과 원문(raw_content)은 컨텍스트 1024 토큰에
                # 감당이 안 된다. 스니펫만 받는다.
                "include_answer": False,
                "include_raw_content": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        return HTTPRequest(
            url="https://api.tavily.com/search",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            body=body,
            timeout=DEFAULT_TIMEOUT,
        )

    def parse(self, payload: dict) -> list[SearchResult]:
        if not isinstance(payload.get("results"), list):
            raise MalformedResponseError("tavily: results가 없다")
        return self._collect(payload["results"], "url", "content")


class SerperProvider(Provider):
    def __init__(self):
        super().__init__("serper", ("SERPER_API_KEY",))

    def build(self, query: str, count: int, api_key: str) -> HTTPRequest:
        body = json.dumps(
            {"q": query, "num": min(max(count, 1), 20)}, ensure_ascii=False
        ).encode("utf-8")
        return HTTPRequest(
            url="https://google.serper.dev/search",
            method="POST",
            headers={"Content-Type": "application/json", "X-API-KEY": api_key},
            body=body,
            timeout=DEFAULT_TIMEOUT,
        )

    def parse(self, payload: dict) -> list[SearchResult]:
        if not isinstance(payload.get("organic"), list):
            raise MalformedResponseError("serper: organic이 없다")
        return self._collect(payload["organic"], "link", "snippet")


PROVIDERS = {
    "brave": BraveProvider,
    "tavily": TavilyProvider,
    "serper": SerperProvider,
}


# ---------------------------------------------------------------- HTTP 계층


def urllib_fetch(req: HTTPRequest) -> HTTPResponse:
    """기본 fetch. 상태 코드는 예외로 던지지 않고 그대로 돌려준다.

    urllib은 4xx/5xx에서 HTTPError를 던지는데, 그러면 429와 진짜 네트워크
    장애가 같은 자리에서 잡혀 구분이 흐려진다. 여기서 상태 코드로 되돌린다.
    """
    r = urllib.request.Request(
        req.url, data=req.body, headers=req.headers, method=req.method
    )
    try:
        with urllib.request.urlopen(r, timeout=req.timeout) as resp:
            return HTTPResponse(resp.status, resp.read())
    except urllib.error.HTTPError as e:
        return HTTPResponse(e.code, e.read() or b"")
    except socket.timeout as e:
        raise SearchTimeout(f"타임아웃 {req.timeout}s: {req.url}") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise SearchTimeout(f"타임아웃 {req.timeout}s: {req.url}") from e
        raise NetworkError(f"네트워크 실패: {e.reason}") from e


# ----------------------------------------------------------------- 클라이언트


@dataclass
class SearchClient:
    provider: Provider
    api_key: str = ""
    fetch: object = None
    timeout: float = DEFAULT_TIMEOUT
    env: dict = field(default_factory=lambda: os.environ)

    def __post_init__(self):
        if self.fetch is None:
            self.fetch = urllib_fetch
        if not self.api_key:
            for k in self.provider.env_keys:
                v = (self.env.get(k) or "").strip()
                if v:
                    self.api_key = v
                    break
        if not self.api_key:
            raise MissingAPIKeyError(
                f"{self.provider.name}: API 키가 없다. "
                f"환경변수 {' 또는 '.join(self.provider.env_keys)}를 설정해라"
            )

    def search(self, query: str, count: int = DEFAULT_COUNT) -> list[SearchResult]:
        q = (query or "").strip()
        if not q:
            raise ValueError("빈 질의로 검색할 수 없다")

        req = self.provider.build(q, count, self.api_key)
        req.timeout = self.timeout
        try:
            resp = self.fetch(req)
        except SearchError:
            raise
        except TimeoutError as e:
            raise SearchTimeout(f"타임아웃 {self.timeout}s") from e
        except OSError as e:
            raise NetworkError(f"네트워크 실패: {e}") from e

        if resp.status == 429:
            raise RateLimitError(f"{self.provider.name}: 속도 제한(429)")
        if resp.status in (401, 403):
            raise AuthError(f"{self.provider.name}: 인증 실패({resp.status})")
        if resp.status != 200:
            raise ProviderError(
                f"{self.provider.name}: HTTP {resp.status} "
                f"{_head(resp.body)}"
            )

        try:
            payload = json.loads(resp.body.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise MalformedResponseError(
                f"{self.provider.name}: JSON 파싱 실패 {_head(resp.body)}"
            ) from e
        if not isinstance(payload, dict):
            raise MalformedResponseError(
                f"{self.provider.name}: 최상위가 객체가 아니다 ({type(payload).__name__})"
            )
        return self.provider.parse(payload)[:count]


def _head(body: bytes, n: int = 120) -> str:
    return body[:n].decode("utf-8", errors="replace")


def make_client(name: str, **kw) -> SearchClient:
    """제공자 이름으로 클라이언트를 만든다."""
    if name not in PROVIDERS:
        raise ValueError(f"모르는 제공자: {name} (가능: {sorted(PROVIDERS)})")
    return SearchClient(provider=PROVIDERS[name](), **kw)
