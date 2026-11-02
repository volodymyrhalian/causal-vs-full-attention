# Як завантажити Causal vs Full Attention на GitHub (Windows)

Проєкт уже готовий: `README.md`, 12 тестів, 18 задокументованих навчань, 6 навчених моделей, виконаний ноутбук, графіки та код. **Не потрібно повторно навчати моделі перед завантаженням.**

## Варіант 1 — через браузер (як минулого разу)

1. Завантаж ZIP і розпакуй архів через **Extract All / Видобути все**.
2. Відкрий https://github.com/new у браузері.
3. Назви репозиторій **`causal-vs-full-attention`** (Public).
4. Description (скопіюй): **`Controlled PyTorch study of causal vs bidirectional Transformer attention, with 18 reproducible runs, trained checkpoints, leakage tests, and visualizations.`**
5. Не додавай новий GitHub README та `.gitignore`: вони вже є всередині архіву.
6. Натисни **Create repository → uploading an existing file**.
7. У Провіднику відкрий **саму розпаковану папку `causal-vs-full-attention`**, виділи її вміст через **Ctrl + A** і перетягни всі файли та папки у браузер. Папка `.github` та файл `.gitignore` мають теж потрапити до списку.
8. У Commit message введи `Initial commit — Causal vs Full Attention Research Lab`, далі натисни **Commit changes**.
9. Перевір, щоб у корені репозиторію були `README.md`, `src`, `models`, `notebooks`, `assets`, `results`, `tests`, `docs`.

**Готова адреса після публікації:** https://github.com/volodymyrhalian/causal-vs-full-attention

### Порада

Не завантажуй ZIP як один файл у репозиторій: потрібен **вміст розпакованої папки**, інакше код та README не відображатимуться правильно.

## Варіант 2 — через Git у PowerShell (зручніше для майбутніх оновлень)

Потрібен Git for Windows, встановлений на комп'ютері. Запусти PowerShell у папці проєкту. Команди:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial commit - Causal vs Full Attention"
git remote add origin https://github.com/volodymyrhalian/causal-vs-full-attention.git
git push -u origin main
```

Якщо Git запитає твою особу, настрой `git config --global user.name "Volodymyr Halian"` та `git config --global user.email "volodiahalian@gmail.com"`. Для авторизації GitHub використовуй браузер/Git Credential Manager, а не пароль акаунта у командному рядку.

## Після завантаження

- Відкрий `README.md` на головній сторінці репозиторію — там має бути графік із результатами.
- Відкрий `notebooks/attention_explorer.ipynb` — GitHub покаже вже виконаний інтерактивний дослід із результатами.
- У About додай Topics: `pytorch`, `transformers`, `attention-mechanism`, `machine-learning`, `deep-learning`, `research`, `nlp`.
- Закріпи (Pin) новий репозиторій у своєму профілі GitHub поруч зі `sound-event-detection-mlpc`.
- Після цього додай посилання на GitHub-проєкт у розділ Projects на своєму CV-сайті.

**Наукова чесність:** у README написано, що це новий незалежний synthetic benchmark, а не раніше виконаний університетський звіт; ці результати не стосуються реального NLP-датасету.
