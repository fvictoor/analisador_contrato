from typing import List, Dict
from datetime import datetime, timedelta
import uuid
from urllib.parse import quote_plus


def _parse_date_iso(date_iso: str) -> datetime:
    return datetime.strptime(date_iso, "%Y-%m-%d")


def _to_all_day_range(date: datetime):
    start = date.strftime("%Y%m%d")
    end = (date + timedelta(days=1)).strftime("%Y%m%d")
    return start, end


def make_google_links_from_dates(
    datas_vencimento: List[Dict], titulo_base: str = "Vencimento de contrato", detalhes: str = ""
) -> List[Dict]:
    links = []
    for item in datas_vencimento:
        date_iso = item.get("data_iso")
        if not date_iso:
            # pular itens sem data
            continue
        descricao = item.get("descricao")
        title = f"{titulo_base}" + (f" - {descricao}" if descricao else "")
        start, end = _to_all_day_range(_parse_date_iso(date_iso))
        text = quote_plus(title)
        details = quote_plus(detalhes)
        url = (
            f"https://calendar.google.com/calendar/render?action=TEMPLATE&text={text}&dates={start}/{end}&details={details}"
        )
        links.append({"descricao": descricao or date_iso, "link": url, "date_iso": date_iso})
    return links


def make_ics_from_dates(
    datas_vencimento: List[Dict], titulo_base: str = "Vencimento de contrato", detalhes: str = ""
) -> str:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//AnalisadorContrato//PT-BR//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]
    for item in datas_vencimento:
        date_iso = item.get("data_iso")
        if not date_iso:
            continue
        descricao = item.get("descricao")
        title = f"{titulo_base}" + (f" - {descricao}" if descricao else "")
        start, end = _to_all_day_range(_parse_date_iso(date_iso))
        uid = str(uuid.uuid4())
        desc = (detalhes or "").replace("\n", "\\n")
        vevent = [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now}",
            f"SUMMARY:{title}",
            f"DESCRIPTION:{desc}",
            f"DTSTART;VALUE=DATE:{start}",
            f"DTEND;VALUE=DATE:{end}",
            "END:VEVENT",
        ]
        lines.extend(vevent)
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)


def make_outlook_links_from_dates(
    datas_vencimento: List[Dict],
    titulo_base: str = "Vencimento de contrato",
    detalhes: str = "",
    timezone: str = "America/Sao_Paulo",
) -> List[Dict]:
    """Gera links de composição de evento para Outlook Web (Live) e Office/M365.
    Usamos evento de dia inteiro quando só há a data.
    """
    links = []
    for item in datas_vencimento:
        date_iso = item.get("data_iso")
        if not date_iso:
            continue
        descricao = item.get("descricao")
        title = f"{titulo_base}" + (f" - {descricao}" if descricao else "")

        start_date = _parse_date_iso(date_iso)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = (start_date + timedelta(days=1)).strftime("%Y-%m-%d")

        subject = quote_plus(title)
        body = quote_plus(detalhes)
        tz = quote_plus(timezone)

        live = (
            "https://outlook.live.com/calendar/0/deeplink/compose?"
            f"allday=true&subject={subject}&body={body}&startdt={start_str}&enddt={end_str}&ctz={tz}"
        )
        office = (
            "https://outlook.office.com/calendar/0/deeplink/compose?"
            f"allday=true&subject={subject}&body={body}&startdt={start_str}&enddt={end_str}&ctz={tz}"
        )

        links.append({
            "descricao": descricao or date_iso,
            "live": live,
            "office": office,
            "date_iso": date_iso,
        })
    return links