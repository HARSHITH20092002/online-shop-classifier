import json
import re
from urllib.parse import urlparse

class DualShopClassifier:
    """
    Cascading Classification Engine:
    - Solution 1: Deterministic Heuristic Guard (Schema.org JSON-LD, Subpaths, Exclusions)
    - Solution 2: Probabilistic Machine Learning Detective (Multilingual Token Scoring)
    """
    def __init__(self):
        self.excluded_platforms = [
            "linkedin.com", "facebook.com", "twitter.com", "x.com", 
            "instagram.com", "github.com", "stackoverflow.com", "wikipedia.org"
        ]
        self.marketplace_subpaths = [
            r"/marketplace", r"/shop", r"/shopping", r"/store", r"/buy", r"/catalog"
        ]
        self.shop_keywords = {
            "en": ["cart", "add to cart", "basket", "checkout", "buy now", "free shipping", "shopping cart", "order online"],
            "fr": ["panier", "ajouter au panier", "commander", "livraison", "frais de port", "tva", "boutique", "mon panier"],
            "de": ["warenkorb", "in den warenkorb", "kasse", "jetzt kaufen", "versandkosten", "inkl. mwst", "bestellen"],
            "es": ["añadir al carrito", "cesta", "comprar ahora", "gastos de envío", "mi carrito", "precio"],
            "it": ["carrello", "aggiungi al carrello", "cassa", "spedizione", "acquista ora", "mio carrello"]
        }
        self.non_shop_text_keywords = [
            "flipbook", "pdf reader", "publish interactive", "digital magazine", 
            "free trial", "annual subscription", "domain for sale"
        ]

    def _extract_schema_ld_json(self, raw_html):
        """Validates embedded Schema.org e-commerce structured objects."""
        if not raw_html or 'application/ld+json' not in raw_html.lower():
            return None
        try:
            matches = re.findall(
                r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', 
                raw_html, 
                re.DOTALL | re.IGNORECASE
            )
            for m in matches:
                clean = m.strip()
                if not clean:
                    continue
                try:
                    data = json.loads(clean)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if isinstance(item, dict):
                            t = str(item.get('@type', '')).lower()
                            if t in ['product', 'offer', 'aggregateoffer', 'store', 'onlinestore']:
                                return f"Schema.org JSON-LD confirmed (@type: {t})"
                except Exception:
                    for t in ['"product"', '"offer"', '"aggregateoffer"', '"store"']:
                        if t in clean.lower():
                            return f"Schema.org pattern match ({t})"
        except Exception:
            pass
        return None

    def solution_1_heuristics(self, data):
        url = data.get("url", "").lower()
        text = data.get("raw_text", "").lower()
        source = data.get("data_source", "Live DOM")
        
        parsed = urlparse(url)
        hostname = parsed.netloc.replace("www.", "")
        path = parsed.path

        # 1. Social Media & Platform Subpath Evaluation
        is_social = any(p in hostname for p in self.excluded_platforms)
        has_subpath = any(re.search(p, path) for p in self.marketplace_subpaths)
        
        if is_social and has_subpath:
            return {
                "is_shop": True, 
                "confidence": 0.95, 
                "method": "Solution 1 (Platform Marketplace)",
                "reason": f"Platform root '{hostname}' contained verified commercial subpath '{path}'."
            }
        if is_social:
            return {
                "is_shop": False, 
                "confidence": 0.95, 
                "method": "Solution 1 (Platform Exclusion)",
                "reason": f"Domain matches non-commercial platform '{hostname}' without active shop subpath."
            }

        # 2. Document Reader / SaaS Service Exclusions
        for non_kw in self.non_shop_text_keywords:
            if non_kw in text:
                return {
                    "is_shop": False, 
                    "confidence": 0.85, 
                    "method": "Solution 1 (Content Exclusion)",
                    "reason": f"Excluded due to non-shop service indicator: '{non_kw}' identified in {source}."
                }

        # 3. Schema.org JSON-LD Verification
        schema_found = self._extract_schema_ld_json(data.get("raw_text", ""))
        if schema_found:
            return {
                "is_shop": True, 
                "confidence": 0.98, 
                "method": "Solution 1 (Schema.org JSON-LD)",
                "reason": f"Embedded e-commerce microdata verified: {schema_found} in {source}."
            }

        # 4. Multilingual Intent Token Verification
        matched_tokens = []
        for lang, keywords in self.shop_keywords.items():
            for kw in keywords:
                if kw in text:
                    matched_tokens.append(f"{kw} [{lang.upper()}]")

        if len(matched_tokens) >= 2 or (source == "Historical Search Snippet" and len(matched_tokens) >= 1):
            return {
                "is_shop": True, 
                "confidence": 0.95, 
                "method": f"Solution 1 (Multi-Lingual Heuristic via {source})",
                "reason": f"Matched commercial purchase tokens: {', '.join(matched_tokens[:4])}."
            }

        return None

    def solution_2_ml_engine(self, data):
        text = data.get("raw_text", "").lower()
        source = data.get("data_source", "Live DOM")
        if not text:
            return {
                "is_shop": False, 
                "confidence": 0.0, 
                "method": "Failed Analysis",
                "reason": "No accessible content found via live requests, archive snapshots, or historical metadata."
            }

        high_intent = [
            "add to cart", "ajouter au panier", "in den warenkorb", 
            "añadir al carrito", "checkout", "panier", "warenkorb", "cesta"
        ]
        general_terms = [
            "price", "prix", "preis", "shipping", "livraison", 
            "versand", "boutique", "shop", "store", "product", "produit"
        ]

        hi_count = sum(1 for term in high_intent if term in text)
        gen_count = sum(1 for term in general_terms if term in text)

        score = min(0.95, (hi_count * 0.40) + (gen_count * 0.15))
        is_shop = score >= 0.50

        return {
            "is_shop": is_shop,
            "confidence": round(score if is_shop else (1.0 - score), 2),
            "method": f"Solution 2 (Probabilistic Scoring via {source})",
            "reason": f"Derived probability score ({score:.2f}) from {hi_count} high-intent and {gen_count} general retail terms."
        }

    def predict(self, page_data):
        if not page_data or not isinstance(page_data, dict):
            page_data = {"raw_text": "", "url": "", "data_source": "Invalid Input"}

        res = self.solution_1_heuristics(page_data)
        if not res:
            res = self.solution_2_ml_engine(page_data)

        res["result"] = "SHOP" if res["is_shop"] else "NOT A SHOP"
        return res

ShopClassifier = DualShopClassifier