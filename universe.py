"""
Built-in lists of each market's main companies (ticker -> name).

Used when Yahoo's market scanner (screener) refuses requests, e.g. "HTTP Error 401":
the Pick Companies tick list, Big Movers and the whole-market scan then use these.
Shares that are renamed or delisted are simply skipped.
"""

US_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon", "GOOGL": "Alphabet (Google)",
    "META": "Meta Platforms", "TSLA": "Tesla", "AVGO": "Broadcom", "BRK-B": "Berkshire Hathaway", "LLY": "Eli Lilly",
    "JPM": "JPMorgan Chase", "V": "Visa", "UNH": "UnitedHealth", "XOM": "Exxon Mobil", "MA": "Mastercard",
    "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble", "HD": "Home Depot", "COST": "Costco", "ABBV": "AbbVie",
    "WMT": "Walmart", "NFLX": "Netflix", "BAC": "Bank of America", "CRM": "Salesforce", "ORCL": "Oracle",
    "KO": "Coca-Cola", "CVX": "Chevron", "MRK": "Merck", "AMD": "Advanced Micro Devices", "PEP": "PepsiCo",
    "ADBE": "Adobe", "TMO": "Thermo Fisher", "LIN": "Linde", "ACN": "Accenture", "MCD": "McDonald's",
    "CSCO": "Cisco", "ABT": "Abbott", "WFC": "Wells Fargo", "DIS": "Walt Disney", "INTU": "Intuit",
    "QCOM": "Qualcomm", "IBM": "IBM", "GE": "GE Aerospace", "CAT": "Caterpillar", "TXN": "Texas Instruments",
    "AMGN": "Amgen", "VZ": "Verizon", "PFE": "Pfizer", "NOW": "ServiceNow", "PM": "Philip Morris",
    "ISRG": "Intuitive Surgical", "UBER": "Uber", "CMCSA": "Comcast", "AMAT": "Applied Materials",
    "GS": "Goldman Sachs", "NEE": "NextEra Energy", "RTX": "RTX", "SPGI": "S&P Global", "T": "AT&T",
    "HON": "Honeywell", "LOW": "Lowe's", "UNP": "Union Pacific", "BKNG": "Booking Holdings", "PGR": "Progressive",
    "MS": "Morgan Stanley", "BLK": "BlackRock", "SYK": "Stryker", "ELV": "Elevance Health", "LMT": "Lockheed Martin",
    "COP": "ConocoPhillips", "DE": "Deere", "SCHW": "Charles Schwab", "TJX": "TJX", "MDT": "Medtronic",
    "PLD": "Prologis", "ADP": "ADP", "CB": "Chubb", "MU": "Micron Technology", "LRCX": "Lam Research",
    "PANW": "Palo Alto Networks", "ADI": "Analog Devices", "MMC": "Marsh McLennan", "BMY": "Bristol-Myers Squibb",
    "GILD": "Gilead Sciences", "SBUX": "Starbucks", "VRTX": "Vertex", "REGN": "Regeneron", "C": "Citigroup",
    "KLAC": "KLA", "SO": "Southern Co", "MO": "Altria", "CI": "Cigna", "ANET": "Arista Networks", "INTC": "Intel",
    "SNPS": "Synopsys", "CDNS": "Cadence Design", "PLTR": "Palantir", "SMCI": "Super Micro Computer",
    "COIN": "Coinbase", "MRVL": "Marvell", "CRWD": "CrowdStrike", "SHOP": "Shopify", "XYZ": "Block",
    "PYPL": "PayPal", "ABNB": "Airbnb", "DELL": "Dell", "ARM": "Arm Holdings", "SNOW": "Snowflake",
    "DDOG": "Datadog", "NKE": "Nike", "BA": "Boeing", "F": "Ford", "GM": "General Motors", "RIVN": "Rivian",
    "SOFI": "SoFi", "HOOD": "Robinhood", "MSTR": "Strategy (MicroStrategy)", "RBLX": "Roblox",
    "DKNG": "DraftKings", "TGT": "Target", "CVS": "CVS Health", "WBD": "Warner Bros. Discovery",
    "APP": "AppLovin", "AXON": "Axon", "TTD": "The Trade Desk", "ZS": "Zscaler", "NET": "Cloudflare",
    "MDB": "MongoDB", "TEAM": "Atlassian", "WDAY": "Workday", "FTNT": "Fortinet", "OKTA": "Okta",
    "DASH": "DoorDash", "LULU": "Lululemon", "ON": "ON Semiconductor", "NXPI": "NXP Semiconductors",
    "MCHP": "Microchip", "TSM": "TSMC", "ASML": "ASML", "BABA": "Alibaba", "PDD": "PDD (Temu)", "JD": "JD.com",
    "NIO": "NIO", "SE": "Sea Ltd", "MELI": "MercadoLibre", "SPOT": "Spotify", "SPY": "S&P 500 ETF (SPY)",
    "QQQ": "Nasdaq-100 ETF (QQQ)", "IWM": "Russell 2000 ETF (IWM)", "TQQQ": "ProShares UltraPro QQQ (3x)",
    "SQQQ": "ProShares UltraPro Short QQQ", "SOXL": "Direxion Semis Bull 3x", "SOXS": "Direxion Semis Bear 3x",
}

ASX_NAMES = {
    "BHP": "BHP Group", "CBA": "Commonwealth Bank", "CSL": "CSL", "NAB": "National Australia Bank",
    "WBC": "Westpac", "ANZ": "ANZ Group", "WES": "Wesfarmers", "MQG": "Macquarie Group", "WDS": "Woodside Energy",
    "FMG": "Fortescue", "GMG": "Goodman Group", "TLS": "Telstra", "RIO": "Rio Tinto", "WOW": "Woolworths",
    "TCL": "Transurban", "ALL": "Aristocrat Leisure", "REA": "REA Group", "STO": "Santos", "QBE": "QBE Insurance",
    "COL": "Coles", "SUN": "Suncorp", "XRO": "Xero", "RMD": "ResMed", "NST": "Northern Star Resources",
    "SHL": "Sonic Healthcare", "ORG": "Origin Energy", "CPU": "Computershare", "ASX": "ASX Ltd",
    "IAG": "Insurance Australia Group", "BXB": "Brambles", "MIN": "Mineral Resources", "PLS": "Pilbara Minerals",
    "S32": "South32", "AMC": "Amcor", "EVN": "Evolution Mining", "TWE": "Treasury Wine Estates", "QAN": "Qantas",
    "APA": "APA Group", "SCG": "Scentre Group", "AGL": "AGL Energy", "ORI": "Orica", "CAR": "CAR Group",
    "SEK": "Seek", "WTC": "WiseTech Global", "NXT": "NextDC", "PME": "Pro Medicus", "LYC": "Lynas Rare Earths",
    "JBH": "JB Hi-Fi", "HVN": "Harvey Norman", "IGO": "IGO", "A2M": "a2 Milk", "REH": "Reece",
    "MPL": "Medibank", "SGP": "Stockland", "DXS": "Dexus", "GPT": "GPT Group", "VCX": "Vicinity Centres",
    "TPG": "TPG Telecom", "CHC": "Charter Hall", "BEN": "Bendigo and Adelaide Bank", "BOQ": "Bank of Queensland",
    "SOL": "Soul Patts", "NHF": "nib Holdings", "ALX": "Atlas Arteria", "AZJ": "Aurizon", "ANN": "Ansell",
    "COH": "Cochlear", "IEL": "IDP Education", "DMP": "Domino's Pizza Enterprises", "FLT": "Flight Centre",
    "WEB": "Web Travel Group", "ZIP": "Zip Co", "TLX": "Telix Pharmaceuticals", "PDN": "Paladin Energy",
    "BOE": "Boss Energy", "DEG": "De Grey Mining", "CMM": "Capricorn Metals", "LTR": "Liontown Resources",
    "SFR": "Sandfire Resources", "RRL": "Regis Resources", "GOR": "Gold Road Resources", "VUL": "Vulcan Energy",
    "ILU": "Iluka Resources", "NIC": "Nickel Industries", "WHC": "Whitehaven Coal", "YAL": "Yancoal",
    "NEC": "Nine Entertainment", "SDF": "Steadfast", "ALQ": "ALS", "ALD": "Ampol", "VEA": "Viva Energy",
}

NZX_NAMES = {
    "FPH": "Fisher & Paykel Healthcare", "AIA": "Auckland Airport", "MEL": "Meridian Energy",
    "IFT": "Infratil", "SPK": "Spark New Zealand", "CEN": "Contact Energy", "AIR": "Air New Zealand",
    "EBO": "EBOS Group", "ATM": "a2 Milk", "FBU": "Fletcher Building", "CNU": "Chorus", "MFT": "Mainfreight",
    "MCY": "Mercury NZ", "GMT": "Goodman Property Trust", "PCT": "Precinct Properties", "KPG": "Kiwi Property",
    "ARG": "Argosy Property", "SKC": "SkyCity Entertainment", "SUM": "Summerset Group", "RYM": "Ryman Healthcare",
    "POT": "Port of Tauranga", "CHI": "Channel Infrastructure", "FRW": "Freightways", "GNE": "Genesis Energy",
    "HGH": "Heartland Group", "IPL": "Investore Property", "KMD": "KMD Brands", "NZX": "NZX Ltd",
    "PFI": "Property for Industry", "SAN": "Sanford", "SKT": "SKY Network TV", "SKL": "Skellerup",
    "TRA": "Turners Automotive", "VCT": "Vector", "VHP": "Vital Healthcare Property", "WIN": "Winton Land",
    "ZEL": "Z Energy", "VGL": "Vista Group", "SPG": "Stride Property", "THL": "Tourism Holdings",
    "OCA": "Oceania Healthcare", "PEB": "Pacific Edge", "GTK": "Gentrack", "SML": "Synlait Milk",
    "TWR": "The Warehouse Group", "NPH": "Napier Port", "MHJ": "Michael Hill", "DGL": "Delegat Group",
    "BGP": "Briscoe Group", "AFI": "Australian Foundation Investment", "HLG": "Hallenstein Glasson",
    "ERD": "EROAD", "VNT": "Ventia", "SCL": "Scales Corporation", "MPG": "Metro Performance Glass",
}

US = list(US_NAMES)
ASX = [s + ".AX" for s in ASX_NAMES]
NZX = [s + ".NZ" for s in NZX_NAMES]

FALLBACK = {"USA": US, "Australia (ASX)": ASX, "New Zealand (NZX)": NZX}
NAMES = {**US_NAMES, **{k + ".AX": v for k, v in ASX_NAMES.items()}, **{k + ".NZ": v for k, v in NZX_NAMES.items()}}


def name_of(symbol: str) -> str:
    return NAMES.get(symbol, symbol)
