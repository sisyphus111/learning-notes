# 网站等宽字体资产

## JetBrains Mono：代码与行内代码

代码使用 [JetBrains Mono v2.304](https://github.com/JetBrains/JetBrainsMono/tree/v2.304)，包含 Regular、Bold、Italic 和 BoldItalic 四个原版 WOFF2 文件，直接从上游 `fonts/webfonts/` 目录获取。字体随网站提供，无需访问者安装字体或连接第三方字体 CDN。原始许可证为 SIL OFL 1.1，见 [JetBrainsMono-OFL.txt](JetBrainsMono-OFL.txt)。

普通代码优先使用 JetBrains Mono；中文等未覆盖字符由下述 Sarasa Fixed 子集补充。代码关闭编程连字，保持操作符逐字符显示。ASCII 字符图继续整体使用 `Notes Mono`，避免混合字体字宽破坏框线与中文对齐。

## Notes Mono：字符图与中文后备

`sarasa-fixed-sc-regular.woff2` 是 [Sarasa Gothic v1.0.43](https://github.com/be5invis/Sarasa-Gothic/releases/tag/v1.0.43) 中 `SarasaFixedSC-Regular.ttf` 的字符子集，供正文中的等宽文本和字符图使用。CSS 使用别名 `Notes Mono`。原始许可证为 SIL OFL 1.1，见同目录 [LICENSE](LICENSE)；保留原作者署名和字体名称。

上游的 Fixed 字型使用半宽破折号且没有编程连字。实际核对源 TTF 与输出 WOFF2：unitsPerEm 为 1000，ASCII、空格、`─│┐└`、`↓←→↔`、上下标数字及希腊字母的 advance 为 500，中文为 1000。`│` 的 y 范围为 −285 至 965，因此字符图使用 1.25 的行高可以连续连接竖线。

子集收录全部笔记 Markdown 中的字符，以及常用 Latin、Greek、标点、箭头、数学符号、box-drawing 与 block-element 范围。新增笔记字符后可按下列步骤再生。原字体不含 `⟹`（U+27F9），该符号仍由浏览器后备字体显示。

在仓库根目录、已准备好的项目 `.venv` 中运行。生成工具为 `fonttools[woff]`、`brotli` 和 `py7zr`；本次使用版本分别为 4.60.2、1.2.1、1.0.0，不属于网站运行依赖。

```sh
curl -fL -o /tmp/SarasaFixedSC-TTF-1.0.43.7z \
  https://github.com/be5invis/Sarasa-Gothic/releases/download/v1.0.43/SarasaFixedSC-TTF-1.0.43.7z
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
import subprocess
import tempfile
import py7zr
from fontTools import subset
from fontTools.ttLib import TTFont

archive = Path('/tmp/SarasaFixedSC-TTF-1.0.43.7z')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
    '34fdccc2899a1676e5399bf2b966e112eeba42a3a8ad88c7dca668a197461cc3'
)
paths = subprocess.check_output(['git', 'ls-files', '-z', '*.md']).decode().split('\0')[:-1]
text = ''.join(Path(p).read_text() for p in paths if not p.startswith('assets/fonts/'))
codepoints = {ord(c) for c in text if c.isprintable()}
for start, end in [(0x20, 0x180), (0x370, 0x400), (0x2000, 0x20A0),
                   (0x2190, 0x2300), (0x2500, 0x25A0)]:
    codepoints.update(range(start, end))
with tempfile.TemporaryDirectory(prefix='notes-sarasa-') as work:
    with py7zr.SevenZipFile(archive) as source:
        source.extract(path=work, targets=['SarasaFixedSC-Regular.ttf'])
    font = TTFont(Path(work) / 'SarasaFixedSC-Regular.ttf', recalcTimestamp=False)
    options = subset.Options()
    options.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]
    options.name_languages = [0x409]
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(unicodes=codepoints)
    subsetter.subset(font)
    font.flavor = 'woff2'
    font.save('assets/fonts/sarasa-fixed-sc-regular.woff2')
PY
```

下载也可使用上游 README 列出的[清华镜像](https://mirrors.tuna.tsinghua.edu.cn/github-release/be5invis/Sarasa-Gothic/LatestRelease/SarasaFixedSC-TTF-1.0.43.7z)，仍需通过上面的 SHA-256 校验。许可证来自该版本的 [LICENSE](https://github.com/be5invis/Sarasa-Gothic/blob/v1.0.43/LICENSE)。
