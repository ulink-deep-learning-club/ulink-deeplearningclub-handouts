import os
import re
import json
import hashlib
import shutil
import subprocess
import requests
from pathlib import Path
from typing import List, Dict
from sphinx.util import logging

logger = logging.getLogger(__name__)

def get_mime_type_from_url(url):
    try:
        headers = {'User-Agent': 'Sphinx-Contributor-Extension'}
        response = requests.head(url, timeout=5, allow_redirects=True, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if ';' in content_type:
            content_type = content_type.split(';')[0].strip()
        return content_type
    except Exception as e:
        logger.warning(f"Failed to fetch mime type for {url}: {e}")
        return None

def download_url_to(url, dest_path: Path):
    if dest_path.exists():
        return dest_path

    try:
        headers = {'User-Agent': 'Sphinx-Contributor-Extension'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(response.content)
        return dest_path
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return None

def get_github_username(email: str):
    # 注意：未认证的 GitHub API 限制为 60次/小时。
    # 生产环境建议通过环境变量注入 GITHUB_TOKEN
    headers = {'User-Agent': 'Sphinx-Contributor-Extension'}
    token = os.getenv('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'token {token}'

    try:
        response = requests.get(f'https://api.github.com/search/users?q={email}', headers=headers, timeout=5)
        if response.status_code == 403:
            logger.warning(f"GitHub API limit reached for {email}")
            return None
        response.raise_for_status()
        data = response.json()
        if data.get("items"):
            return {
                "name": data["items"][0]["login"],
                "avatar": data["items"][0]["avatar_url"]
            }
    except Exception as e:
        logger.warning(f"GitHub API error for {email}: {e}")
    return None

# --- 核心逻辑类 ---

class ContributorRegistry:
    def __init__(self, cache_dir: Path, static_out_dir: Path):
        self.cache_dir = cache_dir
        self.static_out_dir = static_out_dir # Sphinx 构建输出目录中的 _static/avatars
        self.avatar_cache_dir = cache_dir / "avatars"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.avatar_cache_dir.mkdir(parents=True, exist_ok=True)
        self.static_out_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = cache_dir / "index.json"
        if self.index_path.exists():
            try:
                self.registries = json.loads(self.index_path.read_text(encoding='utf-8'))
            except:
                self.registries = []
        else:
            self.registries = []

    def update_cache(self):
        self.index_path.write_text(json.dumps(self.registries, indent=2), encoding='utf-8')

    def get_avatar_filename(self, key):
        """生成唯一的文件名"""
        return hashlib.md5(key.encode()).hexdigest()

    def get_from_details(self, details: Dict[str, str]):
        email = details["email"]
        # 查找缓存
        cached = next((i for i in self.registries if i["email"] == email), None)

        # 如果缓存有效且本地有文件
        if cached:
            cached_file = self.avatar_cache_dir / cached['avatar_filename']
            if cached_file.exists():
                # 关键步骤：每次构建都需要把缓存的图片复制到 Sphinx 的输出目录
                self._copy_to_build(cached_file)
                return cached

        # 如果没有缓存或失效，重新获取
        if cached:
            self.registries.remove(cached)

        gh_data = get_github_username(email)

        if gh_data:
            name = gh_data["name"]
            url = gh_data["avatar"]
            ext = "jpg" # 简化处理，大多数是 jpg/png
        else:
            # Fallback to Gravatar
            name = details["name"]
            # Gravatar MD5 logic
            gravatar_hash = hashlib.md5(email.strip().lower().encode('utf-8')).hexdigest()
            url = f"https://www.gravatar.com/avatar/{gravatar_hash}?d=identicon"
            ext = "jpg"

        # 下载到缓存目录
        filename = f"{self.get_avatar_filename(email)}.{ext}"
        cache_path = self.avatar_cache_dir / filename
        download_url_to(url, cache_path)

        entry = {
            "name": name,
            "email": email,
            "avatar_filename": filename
        }

        self.registries.append(entry)
        self.update_cache()

        # 复制到构建目录
        if cache_path.exists():
            self._copy_to_build(cache_path)

        return entry

    def _copy_to_build(self, source: Path):
        """将缓存的图片复制到 Sphinx 的 _static 输出目录"""
        dest = self.static_out_dir / source.name
        if not dest.exists():
            shutil.copy2(source, dest)

# --- Git 解析逻辑 (保留你的逻辑) ---

git_log_pattern = re.compile(r'(?P<hash>[a-f0-9]+) - (?P<name>[^<]+) <(?P<email>[^>]+)>, (?P<date>\d{4}-\d{2}-\d{2}) : (?P<message>.+)')

def parse_git_log(log_str: str):
    # 处理可能的引号包裹问题
    clean_str = log_str.strip().strip('"')
    match = git_log_pattern.match(clean_str)
    if match:
        return match.groupdict()
    return None

def get_file_commit_log(file_path: Path):
    if not file_path.exists():
        return []
    try:
        # 使用 git root 确保路径正确
        cmd = [
            'git', 'log',
            '--pretty=format:"%h - %an <%ae>, %ad : %s"',
            '--date=short',
            '--', str(file_path.name) # 这里只传文件名，要在 cwd 下运行
        ]
        # 在文件所在目录运行 git，避免相对路径问题
        result = subprocess.run(
            cmd,
            cwd=file_path.parent,
            text=True,
            capture_output=True,
            encoding='utf-8' # 强制 utf-8 避免 Windows GBK 问题
        )
        if result.returncode != 0:
            logger.warning(f"Git error: {result.stderr}")
            return []

        logs = []
        for line in result.stdout.splitlines():
            parsed = parse_git_log(line)
            if parsed:
                logs.append(parsed)
        return logs
    except Exception as e:
        logger.warning(f"Git execution failed: {e}")
        return []

# --- Sphinx integration ---

def generate_html(contributors, logs, pagename: str):
    if not contributors:
        return ""

    html = """
    <div class="git-contributors" style="margin-top: 4em; padding-top: 1em; border-top: 1px solid #888;">
        <p style="font-weight: bold; margin-bottom: 10px;">贡献者与修订历史</p>
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
    """

    # avatar
    for c in contributors:
        # Note: the relative path from the html file
        img_src = f"_static/avatars/{c['avatar_filename']}"
        name = c['name']
        count = c['commits']
        html += f"""
            <div title="{name} ({count} commits)" style="text-align: center;"><a href="{f'https://github.com/{c['github_account']}'}">
                <img src="{'../'*pagename.count('/') + img_src}" alt="{name}" width="40" height="40" style="border-radius: 50%; object-fit: cover; box-shadow: 0px 0px 2px rgba(0, 0, 0, 0.3), 0px 1px 8px -1px rgba(0, 0, 0, 0.2);">
            </a></div>
        """

    html += "</div>"

    # commit history
    html += """
        <details>
            <summary style="cursor: pointer; font-size: 0.9em;">查看详细修订记录</summary>
            <ul style="list-style: none; padding-left: 0; margin-top: 10px; font-size: 0.85em;">
    """

    for log in logs[:10]: # 10 most recent
        hash_short = log['hash']
        date = log['date']
        msg = log['message']
        author = log['name']
        html += f"""
            <li style="margin-bottom: 4px; border-bottom: 1px dashed #888; padding-bottom: 4px;">
                <code style="background: rgba(128, 128, 128, 0.3); padding: 2px 4px; border-radius: 3px;">{hash_short}</code>
                {date} - <strong>{author}</strong>: {msg}
            </li>
        """

    html += """
            </ul>
        </details>
    </div>
    """
    return html

def html_page_context(app, pagename, _, context, doctree):
    if not doctree:
        return

    # get source path (.rst or .md)
    # source_suffix may be a dict or list
    source_suffixes = []
    if isinstance(app.config.source_suffix, dict):
        source_suffixes = list(app.config.source_suffix.keys())
    elif isinstance(app.config.source_suffix, list):
        source_suffixes = app.config.source_suffix
    else:
        # default .rst
        source_suffixes = ['.rst']

    # find the proper extension of the file
    src_path = None
    for suffix in source_suffixes:
        test_path = Path(app.srcdir) / (pagename + suffix)
        if test_path.exists():
            src_path = test_path
            break

    if not src_path:
        return

    if not src_path.exists():
        return

    # init Registry
    # cache: source/.cache/contributors
    # images: build/html/_static/avatars
    cache_dir = Path(app.srcdir) / ".cache" / "contributors"
    static_out_dir = Path(app.outdir) / "_static" / "avatars"

    registry = ContributorRegistry(cache_dir, static_out_dir)

    logs = get_file_commit_log(src_path)
    if not logs:
        return

    # aggregate contributors
    contributors_map = {}
    for log in logs:
        email = log["email"]
        if email not in contributors_map:
            user_info = registry.get_from_details(log)
            contributors_map[email] = {
                "name": f"{log["name"]} ({user_info['name']})" if user_info["name"] != log["name"] else log["name"],
                "github_account": user_info['name'],
                "avatar_filename": user_info['avatar_filename'],
                "commits": 0
            }
        contributors_map[email]["commits"] += 1

    # sort desc
    sorted_contributors = sorted(contributors_map.values(), key=lambda x: x['commits'], reverse=True)

    contributors_html = generate_html(sorted_contributors, logs, pagename)

    # inject to Template Context
    context['git_contributors'] = contributors_html

    # append to body if body exists
    if 'body' in context:
        context['body'] = context['body'] + contributors_html

def setup(app):
    app.connect('html-page-context', html_page_context)
    return {'version': '1.0', 'parallel_read_safe': True}
