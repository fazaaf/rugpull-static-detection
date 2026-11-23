"""Helper functions for the round-2 revision analyses.

Functions marked VERBATIM are copied unchanged from notebooks/pipeline_final.ipynb so that
re-extracted features are identical to the published ones. Everything else is new.
"""
import os, re, json, gzip, hashlib, subprocess
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------- VERBATIM
SL_COLS = ['total_vulnerabilities','high_severity','medium_severity','low_severity','informational','optimization',
           'has_assembly','has_low_level_calls','has_dead_code','has_shadowing','has_costly_loop','has_reentrancy']
RX_COLS = ['has_mint','has_blacklist','has_pause','has_selfdestruct','has_owner_only','has_fee_modifier',
           'has_max_tx','has_whitelist','has_transfer_lock','has_proxy','has_renounce','has_burn']
ST_COLS = ['func_count','modifier_count','event_count','require_count','code_length']
FEAT = SL_COLS + RX_COLS + ST_COLS
E20 = {'totalSupply','balanceOf','transfer','approve','allowance','transferFrom'}

def unpack(res):
    s = (res.get('SourceCode') or '').strip()
    if not s: return None
    try:
        if s.startswith('{{'):
            j = json.loads(s[1:-1]); files = {p: v['content'] for p, v in j['sources'].items()}
            remaps = j.get('settings', {}).get('remappings', [])
        elif s.startswith('{'):
            j = json.loads(s)
            if 'sources' in j:
                files = {p: v['content'] for p, v in j['sources'].items()}; remaps = j.get('settings', {}).get('remappings', [])
            else:
                files = {p: v['content'] for p, v in j.items()}; remaps = []
        else:
            files = {f"{res.get('ContractName') or 'Contract'}.sol": s}; remaps = []
    except Exception:
        return None
    return files, remaps

def pick_main(files, name):
    if name:
        pat = re.compile(rf'\b(contract|library|interface)\s+{re.escape(name)}\b')
        for p, c in files.items():
            if pat.search(c): return p.lstrip('/')
    return max(files, key=lambda p: len(files[p])).lstrip('/')

def solc_ver(res):
    m = re.search(r'(\d+\.\d+\.\d+)', res.get('CompilerVersion', '') or '')
    return m.group(1) if m else None

def slither_feats(det):
    chk = [x['check'] for x in det]
    n = lambda k: sum(1 for x in det if x.get('impact') == k)
    return dict(total_vulnerabilities=len(det), high_severity=n('High'), medium_severity=n('Medium'),
                low_severity=n('Low'), informational=n('Informational'), optimization=n('Optimization'),
                has_assembly=int('assembly' in chk), has_low_level_calls=int('low-level-calls' in chk),
                has_dead_code=int('dead-code' in chk), has_shadowing=int('shadowing-local' in chk),
                has_costly_loop=int('costly-loop' in chk), has_reentrancy=int(any('reentrancy' in c for c in chk)))

def text_feats(source):
    s = source.lower()
    rx = lambda pat: int(bool(re.search(pat, s)))
    return dict(
        has_mint=rx(r'\bmint\b|\b_mint\b|\bminttoken\b'), has_blacklist=rx(r'\bblacklist\b|\bblocklist\b|\bblocked\b'),
        has_pause=rx(r'\bpause\b|\bpaused\b|\bunpause\b'), has_selfdestruct=rx(r'\bselfdestruct\b|\bsuicide\b'),
        has_owner_only=rx(r'\bonlyowner\b|\brequire.*owner\b'), has_fee_modifier=rx(r'\btaxfee\b|\bliquidityfee\b|\b_taxfee\b|\bsetfee\b'),
        has_max_tx=rx(r'\bmaxtxamount\b|\b_maxtx\b|\bmaxwallet\b'), has_whitelist=rx(r'\bwhitelist\b|\bexcluded\b|\b_isexcluded\b'),
        has_transfer_lock=rx(r'\blocktime\b|\btimelock\b|\blocked\b'), has_proxy=rx(r'\bdelegatecall\b|\bproxy\b|\bupgradeable\b'),
        has_renounce=rx(r'\brenounceownership\b'), has_burn=rx(r'\bburn\b|\b_burn\b|\bburntoken\b'),
        func_count=len(re.findall(r'\bfunction\b', s)), modifier_count=len(re.findall(r'\bmodifier\b', s)),
        event_count=len(re.findall(r'\bevent\b', s)), require_count=len(re.findall(r'\brequire\b', s)),
        code_length=len(s))

def solc_minor(res):
    m = re.search(r'0\.(\d+)\.\d+', res.get('CompilerVersion', '') or '')
    return int(m.group(1)) if m else np.nan

def abi_names(res):
    try: return {x.get('name') for x in json.loads(res.get('ABI') or '[]') if x.get('type') == 'function'}
    except Exception: return set()

# ----------------------------------------------------------------------------- loading
def load_res(raw_dir, a):
    p = f'{raw_dir}/{a}.json'
    return json.load(open(p)) if os.path.exists(p) else None

def eff_res(raw_dir, a):
    """Same proxy rule as the pipeline: analyse the implementation if it is verified."""
    res = load_res(raw_dir, a)
    if res is None: return None, 0
    if res.get('Proxy') == '1' and res.get('Implementation'):
        r2 = load_res(raw_dir, res['Implementation'].lower())
        if r2 and r2.get('SourceCode'): return r2, 1
    return res, 0

def write_tree(root, addr, files):
    d = os.path.join(root, addr)
    for p, c in files.items():
        fp = os.path.normpath(os.path.join(d, p.lstrip('/')))
        if not fp.startswith(d): continue
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, 'w', encoding='utf-8') as fh: fh.write(c)
    return d

def run_slither_json(d, main, ver, remaps, solc_path=None, timeout=600):
    """Like run_slither in the pipeline, but returns the full detector list so it can be stored."""
    if not ver: return None, 'no_version'
    env = {**os.environ, 'SOLC_VERSION': ver}
    cmd = ['slither', main, '--json', '-'] + (['--solc-remaps', ' '.join(remaps)] if remaps else [])
    if solc_path: cmd += ['--solc', solc_path]
    try:
        r = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, timeout=timeout)
        data = json.loads(r.stdout) if r.stdout.strip() else None
        if data and data.get('success') is not False and 'results' in data:
            return data['results'].get('detectors', []), 'ok'
        return None, 'slither_error'
    except subprocess.TimeoutExpired:
        return None, 'timeout'
    except Exception:
        return None, 'exception'

def save_json_gz(obj, path):
    with gzip.open(path, 'wt', encoding='utf-8') as fh: json.dump(obj, fh)

def load_json_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as fh: return json.load(fh)

# ----------------------------------------------------------------------------- source handling
def strip_comments(src):
    """Blank out // and /* */ comments, keeping string literals and newlines (line numbers stay valid)."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c in '"\'':
            q = c; j = i + 1
            while j < n and src[j] != q and src[j] != '\n':
                j += 2 if src[j] == '\\' else 1
            out.append(src[i:j + 1]); i = j + 1
        elif src.startswith('//', i):
            j = src.find('\n', i); j = n if j < 0 else j
            out.append(' ' * (j - i)); i = j
        elif src.startswith('/*', i):
            j = src.find('*/', i + 2); j = n if j < 0 else j + 2
            out.append(re.sub(r'[^\n]', ' ', src[i:j])); i = j
        else:
            out.append(c); i += 1
    return ''.join(out)

def _match_brace(s, start):
    """Index of the brace closing the one at s[start]; skips string literals. s must be comment-free."""
    depth, i, n = 0, start, len(s)
    while i < n:
        c = s[i]
        if c in '"\'':
            q = c; i += 1
            while i < n and s[i] != q and s[i] != '\n':
                i += 2 if s[i] == '\\' else 1
        elif c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: return i
        i += 1
    return -1

def main_contract_body(src_nc, name):
    """Text of `contract <name> ... { ... }` (comment-free source). Returns (text, first_line) or (None, None)."""
    if not name: return None, None
    m = re.search(rf'\b(?:abstract\s+)?contract\s+{re.escape(name)}\b', src_nc)
    if not m: return None, None
    b = src_nc.find('{', m.end())
    if b < 0: return None, None
    e = _match_brace(src_nc, b)
    if e < 0: return None, None
    return src_nc[m.start():e + 1], src_nc.count('\n', 0, m.start()) + 1

def function_bodies(src_nc, names):
    """Full text of every function whose name is in `names` (comment-free source), with its first line."""
    out = []
    for m in re.finditer(r'\bfunction\s+(\w+)\s*\(', src_nc):
        if m.group(1) not in names: continue
        b = src_nc.find('{', m.end()); semi = src_nc.find(';', m.end())
        if b < 0 or (0 <= semi < b): continue          # declaration without body
        e = _match_brace(src_nc, b)
        if e > 0: out.append((m.group(1), src_nc.count('\n', 0, m.start()) + 1, src_nc[m.start():e + 1]))
    return out

def regex_feats(text):
    """The 12 regex indicators from text_feats, applied to any text."""
    f = text_feats(text); return {k: f[k] for k in RX_COLS}

# ----------------------------------------------------------------------------- duplicates and similarity
_STR = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
_NUM = re.compile(r'\b(?:0x[0-9a-fA-F]+|\d[\d_]*(?:\.\d+)?(?:e\d+)?)\b')
_TOK = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*|==|!=|<=|>=|&&|\|\||\+\+|--|=>|[^\sA-Za-z0-9_$]')

def norm_level1(src):
    """Comments removed, whitespace collapsed."""
    return re.sub(r'\s+', ' ', strip_comments(src)).strip()

def norm_level2(src):
    """Level 1, plus string and numeric literals replaced by placeholders (token name, supply, fees ignored)."""
    s = strip_comments(src); s = _STR.sub('"S"', s); s = _NUM.sub('0', s)
    return re.sub(r'\s+', ' ', s).strip()

def sha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

def shingles(src_norm, k=5):
    toks = _TOK.findall(src_norm)
    if len(toks) < k: toks = toks + [''] * (k - len(toks))
    return {int.from_bytes(hashlib.blake2b(' '.join(toks[i:i + k]).encode(), digest_size=4).digest(), 'little')
            for i in range(len(toks) - k + 1)}

_P = (1 << 61) - 1
def minhash_matrix(sets, n_perm=256, seed=2026):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 1 << 32, n_perm, dtype=np.uint64); b = rng.integers(0, 1 << 32, n_perm, dtype=np.uint64)
    H = np.empty((len(sets), n_perm), dtype=np.uint64)
    for i, s in enumerate(sets):
        x = np.fromiter(s, dtype=np.uint64, count=len(s))
        H[i] = ((a[:, None] * x[None, :] + b[:, None]) % _P).min(axis=1)
    return H

def est_jaccard(H, rows=None, cols=None, chunk=64):
    """Estimated Jaccard between H[rows] and H[cols] (default: all x all)."""
    R = H if rows is None else H[rows]; C = H if cols is None else H[cols]
    out = np.empty((len(R), len(C)), dtype=np.float32)
    for i in range(0, len(R), chunk):
        out[i:i + chunk] = (R[i:i + chunk, None, :] == C[None, :, :]).mean(axis=2)
    return out

class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra
    def labels(self): return np.array([self.find(i) for i in range(len(self.p))])

# ----------------------------------------------------------------------------- modelling (same settings as the paper)
def feature_groups(d):
    key = [c for c in FEAT if c != 'code_length']
    return (d[key].astype(int).astype(str).agg('|'.join, axis=1) + '|' +
            (d['code_length'] // 1000).astype(int).astype(str)).factorize()[0]

def xgb_model(seed):
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, eval_metric='logloss', random_state=seed)

def cv_folds(X, y, g, n_repeats=5, n_splits=5):
    """Fold-level results of the paper's 5x5 StratifiedGroupKFold XGBoost evaluation."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score
    X = pd.DataFrame(X).reset_index(drop=True); rows, oof = [], []
    for rep in range(n_repeats):
        for k, (tr, te) in enumerate(StratifiedGroupKFold(n_splits, shuffle=True, random_state=rep).split(X, y, g)):
            if len(set(y[te])) < 2: continue
            m = xgb_model(rep); m.fit(X.iloc[tr], y[tr], sample_weight=compute_sample_weight('balanced', y[tr]))
            p = m.predict_proba(X.iloc[te])[:, 1]; pr = (p >= 0.5).astype(int)
            rows.append(dict(rep=rep, fold=k, mcc=matthews_corrcoef(y[te], pr), spec=1 - pr[y[te] == 0].mean(),
                             rec=pr[y[te] == 1].mean(), pr_auc_legit=average_precision_score(1 - y[te], 1 - p),
                             roc_auc=roc_auc_score(y[te], p)))
            oof += [dict(rep=rep, i=i, p=pp) for i, pp in zip(te, p)]
    return pd.DataFrame(rows), pd.DataFrame(oof)

def summarise(f):
    return {k: f'{f[k].mean():.3f} ± {f[k].std():.3f}' for k in ['mcc', 'spec', 'rec', 'pr_auc_legit', 'roc_auc']}

def paired_nb(fa, fb, metric='mcc', n_splits=5):
    """Nadeau-Bengio corrected paired t-test, as in the paper. Positive difference: fa better."""
    from scipy import stats
    a = fa.sort_values(['rep', 'fold'])[metric].values; b = fb.sort_values(['rep', 'fold'])[metric].values
    d = a - b; n = len(d); v = d.var(ddof=1)
    if v == 0: return round(d.mean(), 3), 1.0
    t = d.mean() / np.sqrt((1 / n + 1 / (n_splits - 1)) * v)
    return round(d.mean(), 3), float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))

def holm(p):
    p = np.array(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj

# ----------------------------------------------------------------------------- code regions for regex variants
STD_NAMES = {n.lower() for n in '''SafeMath SignedSafeMath Address Context Ownable Ownable2Step Pausable ReentrancyGuard
ERC20 ERC20Detailed ERC20Burnable ERC20Pausable ERC20Capped ERC20Mintable ERC20Permit ERC20Snapshot ERC20Votes
IERC20 IERC20Metadata IERC20Permit ERC165 IERC165 AccessControl AccessControlEnumerable IAccessControl Roles MinterRole
PauserRole WhitelistAdminRole WhitelistedRole Counters Strings ECDSA EIP712 Math SafeERC20 SafeCast EnumerableSet
EnumerableMap Initializable ContextUpgradeable OwnableUpgradeable ERC20Upgradeable PausableUpgradeable
ReentrancyGuardUpgradeable IERC20Upgradeable IERC20MetadataUpgradeable AddressUpgradeable StorageSlot ERC1967Proxy
TransparentUpgradeableProxy Proxy UpgradeableProxy ProxyAdmin IBeacon ERC1967Upgrade IUniswapV2Factory IUniswapV2Pair
IUniswapV2Router01 IUniswapV2Router02 IUniswapV2ERC20 IUniswapV3Factory IWETH Migrations ApproveAndCallFallBack
Owned TokenInterface ERC20Interface ERC20Basic BasicToken StandardToken MintableToken BurnableToken PausableToken
DetailedERC20 Claimable CanReclaimToken HasNoEther HasNoTokens Destructible'''.split()}

def top_level_blocks(src_nc):
    """[(kind, name, start, end)] of top-level contract/library/interface blocks in comment-free source."""
    out, pos = [], 0
    pat = re.compile(r'\b(abstract\s+contract|contract|library|interface)\s+(\w+)')
    while True:
        m = pat.search(src_nc, pos)
        if not m: break
        b = src_nc.find('{', m.end())
        if b < 0: break
        e = _match_brace(src_nc, b)
        if e < 0: break
        out.append((re.sub(r'\s+', ' ', m.group(1)), m.group(2), m.start(), e + 1)); pos = e + 1
    return out

def project_code(src_nc):
    """Contracts written for this token: excludes interfaces, libraries and well-known standard contracts."""
    keep = [src_nc[s:e] for k, n, s, e in top_level_blocks(src_nc) if k != 'interface' and k != 'library'
            and n.lower() not in STD_NAMES]
    return '\n'.join(keep)

_MINT_FN = re.compile(r'\bfunction\s+\w*mint\w*\s*\([^)]*\)[^{;]*\b(public|external)\b', re.I)
def mint_variants(code_nc):
    """Separates an externally callable mint function from a mint call inside the constructor."""
    ctor = ''.join(b for _, _, b in function_bodies(code_nc, {'constructor'}))
    for m in re.finditer(r'\bconstructor\s*\(', code_nc):
        b = code_nc.find('{', m.end()); e = _match_brace(code_nc, b) if b >= 0 else -1
        if e > 0: ctor += code_nc[b:e + 1]
    return dict(has_mint_fn_external=int(bool(_MINT_FN.search(code_nc))),
                has_ctor_mint=int(bool(re.search(r'\b_mint\s*\(', ctor))))
