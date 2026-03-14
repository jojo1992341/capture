#!/usr/bin/env python3
"""
Dofus HDV Screenshot Price Extractor - 100% local
Extrait nom, niveau, type, prix_moyen, prix ×1/×10/×100/×1000.

Usage:
    python dofus_extractor.py image.png [--output resultats.json]
    python dofus_extractor.py dossier/  [--output resultats.json]

Dépendances :
    pip install pytesseract opencv-python pillow
    + Tesseract OCR installé (https://tesseract-ocr.github.io/)
"""

import sys, re, json, argparse, logging, difflib, unicodedata
from pathlib import Path
import cv2, numpy as np, pytesseract

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def setup_log_file(path: str) -> None:
    """Ajoute un FileHandler pour écrire tous les logs dans un fichier."""
    fh = logging.FileHandler(path, encoding="utf-8", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    log.info(f"Logs écrits dans : {path}")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
SCALE       = 3
THRESH_VAL  = 110
VALID_LOTS  = {1, 10, 100, 1000}


# ── Dictionnaire d'items (optionnel) ────────────────────────────
# Chargé via --dict chemin/vers/items.json
# Format : [{"id":..., "name":{"fr":"..."}, "level":..., "type":{"name":{"fr":"..."}}}]
#
# Matching multi-passes (sans doublon) :
#   Passes 1-4   : seuil 100%  |  5-8 : 75%  |  9-12 : 50%  |  13-16 : 25%
#   Pour chaque seuil, 4 variantes :
#     A) type AND niveau AND nom (avec accents)
#     B) type AND niveau AND nom (sans accents)
#     C) (type OR  niveau) AND nom (avec accents)
#     D) (type OR  niveau) AND nom (sans accents)

_ITEM_INDEX   = []    # [(item_dict, norm_with_accents, norm_no_accents)]
_ITEM_INDEX_AVAILABLE = []   # copie en mémoire, réduite au fil des passes

def _normalize(s: str, strip_accents: bool = False) -> str:
    """Minuscules, sans ponctuation. strip_accents=True retire aussi les accents."""
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    if strip_accents:
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    else:
        s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^a-z0-9\u00e0-\u00ff ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def load_item_dict(path: str) -> int:
    """Charge le fichier JSON d'items et construit l'index."""
    global _ITEM_INDEX, _ITEM_INDEX_AVAILABLE
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if isinstance(items, dict):
        items = items.get("items", list(items.values()))
    _ITEM_INDEX = []
    for item in items:
        name_fr = (item.get("name") or {}).get("fr") or ""
        if name_fr:
            _ITEM_INDEX.append((
                item,
                _normalize(name_fr, strip_accents=False),
                _normalize(name_fr, strip_accents=True),
            ))
    _ITEM_INDEX_AVAILABLE = list(_ITEM_INDEX)
    log.info(f"  Dictionnaire chargé : {len(_ITEM_INDEX)} items")
    return len(_ITEM_INDEX)

def _name_score(ocr_norm: str, item_norm: str) -> float:
    return difflib.SequenceMatcher(None, ocr_norm, item_norm).ratio()

def _item_level(item: dict) -> int | None:
    return item.get("level")

def _item_type(item: dict) -> str:
    return _normalize((item.get("type") or {}).get("name", {}).get("fr", ""), strip_accents=False)

def _item_type_na(item: dict) -> str:
    return _normalize((item.get("type") or {}).get("name", {}).get("fr", ""), strip_accents=True)

# Les 20 passes sous forme de config
# (strip_accents_name, require_both_level_and_type, threshold, name_only)
# name_only=True → ignore complètement type et niveau (passes 17-20)
_PASSES = [
    # seuil 100%
    (False, True,  1.00, False),   # 1  : type AND niveau, nom exact
    (True,  True,  1.00, False),   # 2  : type AND niveau, nom exact sans accents
    (False, False, 1.00, False),   # 3  : type OR  niveau, nom exact
    (True,  False, 1.00, False),   # 4  : type OR  niveau, nom exact sans accents
    # seuil 75%
    (False, True,  0.75, False),   # 5
    (True,  True,  0.75, False),   # 6
    (False, False, 0.75, False),   # 7
    (True,  False, 0.75, False),   # 8
    # seuil 50%
    (False, True,  0.50, False),   # 9
    (True,  True,  0.50, False),   # 10
    (False, False, 0.50, False),   # 11
    (True,  False, 0.50, False),   # 12
    # seuil 25%
    (False, True,  0.25, False),   # 13
    (True,  True,  0.25, False),   # 14
    (False, False, 0.25, False),   # 15
    (True,  False, 0.25, False),   # 16
    # nom seul (type et niveau ignorés — filet de sécurité)
    (False, False, 1.00, True),    # 17 : nom exact (avec accents)
    (False, False, 0.75, True),    # 18 : nom ≥ 75%
    (False, False, 0.50, True),    # 19 : nom ≥ 50%
    (False, False, 0.25, True),    # 20 : nom ≥ 25%
]

def run_multipass_matching(ocr_results: list[dict]) -> None:
    """
    Fait correspondre les résultats OCR avec le dictionnaire en 20 passes.
    Modifie ocr_results en place. Chaque item du dict ne peut être assigné qu'une fois.
    """
    if not _ITEM_INDEX_AVAILABLE:
        return

    # Marque ceux déjà assignés (par l'id du dict)
    assigned_dict_ids: set = set()
    # Résultats non encore résolus
    unresolved = list(range(len(ocr_results)))

    for pass_num, (strip_acc, require_both, threshold, name_only) in enumerate(_PASSES, start=1):
        if not unresolved:
            break

        newly_resolved = []

        for idx in unresolved:
            r = ocr_results[idx]
            if r.get("id") is not None:   # déjà résolu dans une passe précédente
                newly_resolved.append(idx)
                continue

            ocr_name  = r.get("nom")
            ocr_level = r.get("niveau")
            ocr_type  = r.get("type")

            # Noms OCR normalisés selon la passe
            norm_ocr = _normalize(ocr_name or "", strip_accents=strip_acc)
            norm_type_ocr = _normalize(ocr_type or "", strip_accents=strip_acc)

            best_item_entry = None
            best_score = -1.0

            for entry in _ITEM_INDEX_AVAILABLE:
                item, norm_acc, norm_na = entry
                item_id = item.get("id")
                if item_id in assigned_dict_ids:
                    continue

                # Vérification niveau (±2 tolérance OCR)
                level_ok = (ocr_level is not None and
                            _item_level(item) is not None and
                            abs(_item_level(item) - ocr_level) <= 2)

                # Vérification type
                norm_itype = _item_type_na(item) if strip_acc else _item_type(item)
                type_ok = bool(norm_type_ocr and norm_itype and norm_type_ocr == norm_itype)

                # Filtre selon le mode de la passe
                if name_only:
                    pass   # pas de filtre niveau/type pour les passes 17-20
                elif require_both:
                    if not (level_ok and type_ok):
                        continue
                else:
                    if not (level_ok or type_ok):
                        continue

                # Score nom
                norm_item = norm_na if strip_acc else norm_acc
                score = _name_score(norm_ocr, norm_item)

                if score > best_score:
                    best_score = score
                    best_item_entry = entry

            if best_item_entry is not None and best_score >= threshold:
                item = best_item_entry[0]
                item_id = item.get("id")
                assigned_dict_ids.add(item_id)

                item_name = (item.get("name") or {}).get("fr", "")
                item_type = (item.get("type") or {}).get("name", {}).get("fr", "")

                if r.get("nom") != item_name:
                    log.info(f"  Dict P{pass_num} [{best_score:.2f}]: "
                             f"'{r.get('nom')}' → '{item_name}'")

                r["nom"]    = item_name or r.get("nom")
                r["type"]   = item_type or r.get("type")
                r["niveau"] = item.get("level") or r.get("niveau")
                r["id"]     = item_id
                newly_resolved.append(idx)

        unresolved = [i for i in unresolved if i not in newly_resolved]

    # Items non résolus → id null
    for idx in unresolved:
        if ocr_results[idx].get("id") is None:
            ocr_results[idx]["id"] = None

def match_item_dict(ocr_name, ocr_level, ocr_type):
    """Compatibilité : utilisé pour le mode image unique (sans multi-passes)."""
    if not _ITEM_INDEX_AVAILABLE:
        return None, 0.0, "no_dict"
    # Pour une seule image, on fait un mini multi-pass sur 1 résultat
    tmp = [{"nom": ocr_name, "niveau": ocr_level, "type": ocr_type, "id": None}]
    run_multipass_matching(tmp)
    if tmp[0].get("id") is not None:
        # Retrouve l'item complet
        for entry in _ITEM_INDEX:
            if entry[0].get("id") == tmp[0]["id"]:
                return entry[0], 1.0, "multipass"
    return None, 0.0, "no_match"

# ── Masquage des icônes kama (pièces d'or) ──────────────────────
# L'icône kama est lue comme "4" par l'OCR → on la masque avant l'OCR.
GOLD_HSV_LOWER = np.array([15, 100, 150])
GOLD_HSV_UPPER = np.array([35, 255, 255])

def mask_kama_icons(img):
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GOLD_HSV_LOWER, GOLD_HSV_UPPER)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    result = img.copy()
    result[mask > 0] = [40, 40, 40]   # remplace par fond sombre
    return result

# ── Corrections OCR des caractères spéciaux ────────────────────
# Tesseract confond systématiquement certains caractères français.
# On corrige le texte brut AVANT le parsing.

OCR_CORRECTIONS = [
    # Œ (OE ligature) — Tesseract lit souvent Œ comme CE, 3, C, G, O, ou rien
    ("CEuf",   "Œuf"),
    ("CEil",   "Œil"),
    ("3uf",    "Œuf"),
    ("3 uf",   "Œuf"),
    ("3il",    "Œil"),
    ("3 il",   "Œil"),
    ("Gil",    "Œil"),
    ("Cil",    "Œil"),
    ("OEuf",   "Œuf"),
    ("OEil",   "Œil"),
    # Quand le Œ disparaît totalement : "uf" ou "il" isolé après séparateur
    (" uf ",   " Œuf "),
    (" il ",   " Œil "),
    ("+ uf",   "+ Œuf"),
    ("• uf",   "• Œuf"),
    ("* uf",   "* Œuf"),
    ("¢ uf",   "¢ Œuf"),
    ("+ il",   "+ Œil"),
    ("• il",   "• Œil"),
    ("* il",   "* Œil"),
    # œ minuscule au milieu d'un mot : Tesseract le lit souvent "ee"
    # Ex: "Supervizœuf" → "Supervizeeuf", "Zœuf" → "Zeeuf"
    ("eeuf",   "œuf"),
    ("ceuf",   "œuf"),   # "Supervizceuf" → "Supervizœuf"
    ("eeil",   "œil"),
    ("ceil",   "œil"),
    ("eeuvre", "œuvre"),
    # Accents circonflexes perdus sur certains mots courants du jeu
    ("Vétement",    "Vêtement"),
    ("vétement",    "vêtement"),
    ("Matériel",    "Matériel"),   # déjà bon normalement, au cas où
    ("Préparation", "Préparation"),
    # à perdu devant article (OCR lit "a" au lieu de "à")
    (" a l'",  " à l'"),
    (" a l’", " à l’"),
    (" a la ", " à la "),
    (" a le ", " à le "),
]

def fix_ocr_text(text: str) -> str:
    """Applique les corrections OCR sur le texte brut du popup."""
    for wrong, correct in OCR_CORRECTIONS:
        text = text.replace(wrong, correct)
    return text


# ── Détection du popup ──────────────────────────────────────────

def find_popup_anchor(img):
    """Localise 'Prix moyen : <nombre>' dans l'image entière."""
    data = pytesseract.image_to_data(
        img, lang="eng", config="--psm 11 --oem 3",
        output_type=pytesseract.Output.DICT,
    )
    n = len(data["text"])
    words = [dict(text=data["text"][i], x=data["left"][i], y=data["top"][i],
                  conf=data["conf"][i]) for i in range(n)]

    candidates = []
    for i, w in enumerate(words):
        if w["conf"] < 5 or "prix" not in w["text"].lower():
            continue
        for j in range(max(0, i - 1), min(n, i + 6)):
            w2 = words[j]
            if "moyen" not in w2["text"].lower():
                continue
            if abs(w2["y"] - w["y"]) > 20 or abs(w2["x"] - w["x"]) > 160:
                continue
            price_val = None
            # Collecte les tokens numériques consécutifs après "moyen"
            # pour gérer les prix avec espaces ("4 437", "17 677", etc.)
            price_parts = []
            for k in range(j, min(n, j + 10)):
                wk = words[k]
                raw = re.sub(r"[^\d]", "", wk["text"])
                if raw and wk["x"] > w2["x"] and abs(wk["y"] - w2["y"]) < 25:
                    price_parts.append(raw)
                elif price_parts:
                    break  # fin de la séquence numérique
            if price_parts:
                price_val = int("".join(price_parts))
            candidates.append(dict(x=w["x"], y=w["y"], prix_moyen=price_val))
            break

    with_price = [c for c in candidates if c["prix_moyen"] is not None]
    return with_price[0] if with_price else (candidates[0] if candidates else None)

# ── Crop + OCR ──────────────────────────────────────────────────

def crop_popup(img, ax, ay):
    h, w = img.shape[:2]
    return img[max(0, ay - 80):min(h, ay + 290),
               max(0, ax - 90):min(w, ax + 330)]

def raw_ocr(crop):
    masked = mask_kama_icons(crop)
    large  = cv2.resize(masked, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    _, th  = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY)
    raw = pytesseract.image_to_string(th, lang="eng", config="--psm 6 --oem 3")
    return fix_ocr_text(raw)

# ── Parsing ─────────────────────────────────────────────────────

def _digit_tokens_pos(line: str):
    """[(token_str, start_pos)] pour chaque séquence de chiffres."""
    return [(m.group(), m.start()) for m in re.finditer(r"\d+", line)]

def _lot_token_to_size(tok: str):
    """Renvoie la taille de lot (1/10/100/1000) ou None."""
    if not tok.lstrip("0"):                      # "0", "00", "000"
        return 1000 if len(tok) == 3 else 100
    n = int(tok)
    return n if n in VALID_LOTS else None

def _reconstruct_price(tokens: list[str]) -> int | None:
    """
    Reconstruit un prix depuis plusieurs tokens.
    Concatène directement (ex: "9" + "995" → 9995, "17" + "677" → 17677).
    Sanity-check à 100M de kamas.
    """
    if not tokens:
        return None
    combined = int("".join(tokens))
    if combined > 100_000_000:
        return max(int(t) for t in tokens if t.isdigit())
    return combined

def extract_lot_prices(lines: list[str]) -> dict:
    prices = {1: None, 10: None, 100: None, 1000: None}

    # Lignes avec ACHETER = lignes du tableau
    acheter_lines = [l for l in lines if "acheter" in l.lower()]
    if not acheter_lines:
        in_table = False
        for line in lines:
            if re.search(r"\blot\b", line, re.I) and re.search(r"\bprix\b", line, re.I):
                in_table = True; continue
            if in_table and re.search(r"\d{2,}", line):
                acheter_lines.append(line)

    row_data = []    # [(lot_size_or_None, price_or_None)]
    # Lots déjà assignés : utilisé pour ignorer les doublons OCR dans une même ligne
    # Ex: "1 10 1 396 ACHETER" → le "1" en tête est un artefact du lot précédent,
    # on doit ignorer lot=1 (déjà vu) et prendre lot=10.
    assigned_so_far = set()

    for line in acheter_lines:
        toks = _digit_tokens_pos(line)
        if not toks:
            continue

        lot_size = None
        lot_end  = None

        # Cherche le premier token lot VALIDE ET NON ENCORE ASSIGNÉ
        for tok, pos in toks:
            ls = _lot_token_to_size(tok)
            if ls is not None and ls not in assigned_so_far:
                lot_size = ls
                lot_end  = pos + len(tok)
                break

        if lot_size is not None and lot_end is not None:
            # Prix = tokens APRÈS le lot (position strictement supérieure)
            price_tokens = [t for t, p in toks if p > lot_end]
            assigned_so_far.add(lot_size)
        else:
            # Lot invisible (lot=1 caché par l'icône) :
            # Le dernier token numérique dans la ligne est le prix.
            all_num = [t for t, _ in toks]
            if not all_num:
                continue
            price_tokens = [all_num[-1]]

        price = _reconstruct_price(price_tokens) if price_tokens else None
        row_data.append([lot_size, price])

    # Post-traitement : affecte le lot manquant aux lignes sans lot
    assigned = {rd[0] for rd in row_data if rd[0] is not None}
    for rd in row_data:
        if rd[0] is None and rd[1] is not None:
            for c in (1, 10, 100, 1000):
                if c not in assigned:
                    rd[0] = c; assigned.add(c); break

    for lot_size, price in row_data:
        if lot_size and price and prices[lot_size] is None:
            prices[lot_size] = price

    return prices

# ── Nom ─────────────────────────────────────────────────────────
SKIP_KW = {"niv", "prix", "lot", "quantit", "acheter", "inventaire",
           "banque", "havre", "vente", "achat", "reinit", "afficher"}

def _is_meaningful_name(s: str) -> bool:
    clean = re.sub(r"[^A-Za-z\u00C0-\u00FFŒœ0-9 '\-]", "", s).strip()
    clean_low = clean.lower()
    # Exige au moins un MOT de 4+ lettres pour rejeter les artefacts OCR courts
    # Ex: "ra ew", "la aw", "aw" → rejetés  |  "Avoine", "Dent de Larve" → acceptés
    has_real_word = bool(re.search(r"[A-Za-z\u00C0-\u00FFŒœ]{3,}", clean))
    return (len(clean) >= 3
            and has_real_word
            and not any(re.search(r"\b" + kw + r"\b", clean_low) for kw in SKIP_KW))

def _clean_name(raw: str) -> str:
    """Extrait le nom propre de la ressource depuis une ligne OCR brute.
    Gère toutes les formes d'apostrophe (droite ' U+0027, courbe \u2019, etc.)
    ainsi que les connecteurs français : de, du, des, d', l', à l', au, aux...
    """
    CHAR = r"[A-Za-z\u00C0-\u00FFŒœ]"
    APOS = r"['\u2018\u2019\u02bc]"   # toutes formes d'apostrophe
    m = re.search(
        r"([A-Z\u00C0-\u00DCŒ]" + CHAR + r"{2,}"        # 1er mot ≥ 3 lettres (majuscule)
        r"(?:[-]" + CHAR + r"+)*"                           # tirets optionnels
        r"(?:\s+"                                           # mots suivants
            r"(?:(?:de |du |des |\u00e0 |au |aux |en )"   # prépositions longues
            r"|(?:d" + APOS + r"|l" + APOS + r"|\u00e0 l" + APOS + r"))?"  # élisions
            + CHAR + r"+"
            r"(?:" + APOS + CHAR + r"+)*"                   # apostrophe interne (l'envers)
            r"(?:[-]" + CHAR + r"+)*"
        r")*)",
        raw
    )
    if m:
        return m.group(1).strip()
    clean = re.sub(r"^[^A-Za-z\u00C0-\u00FFŒœ]+", "", raw).strip()
    return re.sub(r"[^A-Za-z\u00C0-\u00FFŒœ0-9 '\u2019\-]", "", clean).strip()


# ── Parse complet ────────────────────────────────────────────────

def parse_popup(raw_text: str, prix_moyen_hint=None) -> dict:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    result = dict(nom=None, niveau=None, type=None,
                  prix_moyen=prix_moyen_hint,
                  prix_1=None, prix_10=None, prix_100=None, prix_1000=None)

    # Ligne "Niv. XX"
    niv_idx = next((i for i, l in enumerate(lines) if re.search(r"\bNiv\b", l, re.I)), None)

    # ── Nom ──────────────────────────────────────────────────────
    for i in (range(niv_idx - 1, -1, -1) if niv_idx else range(len(lines))):
        if _is_meaningful_name(lines[i]):
            name = _clean_name(lines[i])
            if name:
                result["nom"] = name
            break

    # ── Niveau + Type ────────────────────────────────────────────
    if niv_idx is not None:
        line = lines[niv_idx]
        m = re.search(r"[Nn]iv[.:]?\s*(\d+)", line)
        if m:
            result["niveau"] = int(m.group(1))
            after = re.sub(r"^[^A-Za-zÀ-ÿŒœ]+", "", line[m.end():]).strip()
            # Le type est soit un mot unique capitalisé, soit deux mots liés par
            # une préposition/article connu (ex: "Ressource diverse", "Pierre précieuse")
            # On exclut les doublons OCR ("Os os") en n\'autorisant que les particules connues
            _PARTICLES = r"(?:de|du|des|d\'|d\u2019|la|le|les|en|aux?|et|précieuse|diverse|brute|combat)\s+"
            tm = re.match(
                r"([A-ZÀÂÉÈÊËÎÏÔÙÛÜŒ][A-Za-zÀ-ÿŒœé]+(?:[-][A-Za-zÀ-ÿŒœé]+)*"
                r"(?:\s+(?:de|du|des|d\'|la|le|les|en|aux?|et)\s+[A-Za-zÀ-ÿŒœé]+(?:[-][A-Za-zÀ-ÿŒœé]+)*"
                r"|(?:\s+[A-Za-zÀ-ÿŒœé]{4,}(?:[-][A-Za-zÀ-ÿŒœé]+)*))*)", after)
            if tm:
                result["type"] = tm.group(1).strip()
            else:
                fm = re.search(r"[A-Za-zÀ-ÿ]{3,}", after)
                if fm:
                    result["type"] = fm.group(0)

    # ── Prix moyen : regex directe ───────────────────────────────
    # Gère les prix avec espaces comme "4 437" ou "17 677"
    for line in lines:
        if "moyen" in line.lower():
            m = re.search(r"moyen\s*:?\s*([\d][\d\s]*)", line, re.I)
            if m:
                # Récupère uniquement les chiffres et espaces en tête, stop au premier non-chiffre
                raw = re.match(r"[\d\s]+", m.group(1))
                if raw:
                    result["prix_moyen"] = int(re.sub(r"\s", "", raw.group(0)))
            break

    # ── Table des lots ───────────────────────────────────────────
    lp = extract_lot_prices(lines)
    result["prix_1"]    = lp[1]
    result["prix_10"]   = lp[10]
    result["prix_100"]  = lp[100]
    result["prix_1000"] = lp[1000]

    return result

# ── Pipeline ─────────────────────────────────────────────────────

def imread_unicode(path: str):
    """
    Remplace cv2.imread() pour supporter les chemins avec caractères
    non-ASCII sur Windows (accents, espaces, apostrophes, etc.).
    Lit le fichier en binaire via Python puis décode avec numpy/cv2.
    """
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def process_image(img_path: str) -> dict | None:
    img = imread_unicode(str(img_path))
    if img is None:
        log.warning(f"Impossible de lire : {img_path}"); return None

    log.info(f"Traitement : {Path(img_path).name}")
    anchor = find_popup_anchor(img)
    if anchor is None:
        log.warning("  → Aucun popup 'Prix moyen' détecté"); return None

    log.info(f"  → Ancre ({anchor['x']}, {anchor['y']}), moyen hint={anchor['prix_moyen']}")
    crop    = crop_popup(img, anchor["x"], anchor["y"])
    raw_txt = raw_ocr(crop)
    log.debug("OCR:\n" + raw_txt)

    result = parse_popup(raw_txt, anchor["prix_moyen"])

    result["id"]     = None   # sera rempli par run_multipass_matching
    result["source"] = Path(img_path).name
    log.info(f"  → {result['nom']} | niv.{result['niveau']} {result['type']} | "
             f"moy={result['prix_moyen']} | "
             f"×1={result['prix_1']}  ×10={result['prix_10']}  "
             f"×100={result['prix_100']}  ×1000={result['prix_1000']}")
    return result

def process_folder(folder: str) -> list:
    results = []
    images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not images:
        log.warning(f"Aucune image dans {folder}"); return results
    for p in images:
        r = process_image(str(p))
        if r: results.append(r)
    # Multi-passes sur l'ensemble des résultats (évite les doublons)
    if _ITEM_INDEX_AVAILABLE and results:
        log.info(f"  Matching multi-passes sur {len(results)} résultats...")
        # Réinitialise les ids assignés par process_image (mode image unique)
        for r in results:
            r["id"] = None
        _ITEM_INDEX_AVAILABLE[:] = list(_ITEM_INDEX)  # remet le dict complet
        run_multipass_matching(results)
    return results

# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extracteur de prix HDV Dofus (100% local, Tesseract OCR)"
    )
    parser.add_argument("input",  help="Image ou dossier de captures d'écran")
    from datetime import datetime
    default_output = datetime.now().strftime("Prix%Y-%m-%d_%H-%M-%S.json")
    parser.add_argument("--output", "-o", default=default_output)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dict", "-d", default=None,
                        metavar="ITEMS.JSON",
                        help="Dictionnaire JSON des items Dofus pour corriger les noms OCR")
    parser.add_argument("--logs", "-l", default=None,
                        metavar="FICHIER.TXT",
                        help="Fichier où écrire tous les logs (ex: --logs debug.txt)")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.logs:
        setup_log_file(args.logs)

    if args.dict:
        try:
            load_item_dict(args.dict)
        except Exception as e:
            log.error(f"Impossible de charger le dictionnaire : {e}")

    inp = Path(args.input)
    if not inp.exists():
        log.error(f"Introuvable : {inp}"); sys.exit(1)

    results = process_folder(str(inp)) if inp.is_dir() else (
              [r] if (r := process_image(str(inp))) else [])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log.info(f"\n✅  {len(results)} ressource(s) extraite(s) → {args.output}")
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
