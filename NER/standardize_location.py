import requests
import re

# Normalize input
def normalize_key(text):
    return re.sub(r"[^a-z0-9]", "", text.strip().lower())

# Search for city/place (normal flow)
def get_country_from_geonames(city_name, username="vyphung"):
    url = "http://api.geonames.org/searchJSON"
    params = {
        "q": city_name,
        "maxRows": 1,
        "username": username
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if data.get("geonames"):
            return data["geonames"][0]["countryName"]
    except Exception as e:
        print("GeoNames searchJSON error:", e)
    return None

# Search for country info using alpha-2/3 codes or name
def get_country_from_countryinfo(input_code, username="vyphung"):
    url = "http://api.geonames.org/countryInfoJSON"
    params = {
        "username": username
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if data.get("geonames"):
            input_code = input_code.strip().upper()
            for country in data["geonames"]:
                # Match against country name, country code (alpha-2), iso alpha-3
                if input_code in [
                    country.get("countryName", "").upper(),
                    country.get("countryCode", "").upper(),
                    country.get("isoAlpha3", "").upper()
                ]:
                    return country["countryName"]
    except Exception as e:
        print("GeoNames countryInfoJSON error:", e)
    return None

# Combined smart lookup
def smart_country_lookup(user_input, username="vyphung"):
    raw_input = user_input.strip()
    normalized = re.sub(r"[^a-zA-Z0-9]", "", user_input).upper()  # normalize for codes (no strip spaces!)

    # Special case: if user writes "UK: London" → split and take main country part
    if ":" in raw_input:
        raw_input = raw_input.split(":")[0].strip()  # only take "UK"
    # First try as country code (if 2-3 letters or common abbreviation)
    if len(normalized) <= 3:
      if normalized.upper() in ["UK","U.K","U.K."]:
        country = get_country_from_geonames(normalized.upper(), username=username)
        if country:
          return country
      else:  
        country = get_country_from_countryinfo(raw_input, username=username)
        if country:
            return country
    country = get_country_from_countryinfo(raw_input, username=username)  # try full names
    if country:
        return country
    # Otherwise, treat as city/place
    country = get_country_from_geonames(raw_input, username=username)
    if country:
        return country

    return "Not found"