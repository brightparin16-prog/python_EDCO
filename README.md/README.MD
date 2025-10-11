# 🛠️ Git Command Summary

A colorful guide to common Git commands, with syntax highlighting for easier reference.

---

## 📝 Check Git Version

```bash
git -v
```
| **Checks the installed version of Git.**

---

## 🧑‍💻 Configure Git Credentials

```bash
git config --global user.name "[your GitHub username]"
```
| **Set your GitHub username globally on your computer.**

```bash
git config --global user.email "[your GitHub email]"
```
| **Set your GitHub email globally on your computer.**

```bash
git config --global credential.helper
```
| **Set Git to remember your username and email on your computer.**

```bash
git config --list
```
| **Check your current Git configuration (username & email).**

---

## 📂 Initialize & Add Code to GitHub

```bash
git init
```
| **Initialize a new Git repository in your project.**

```bash
git add .
```
| **Add all files in the project to the staging area.**

```bash
git commit -m "[your message about adding these files]"
```
| **Commit your changes with a descriptive message.**

```bash
git branch -M main
```
| **Rename the current branch to `main`.**

```bash
git remote add origin [your GitHub repo URL]
```
| **Link your local repository to a GitHub repository.**

```bash
git remote -v
```
| **Check the remote repositories linked to your project.**

```bash
git push -u origin main
```
| **Push commits to GitHub and set `origin/main` as the default upstream branch.**

---

<details>
  <summary>🎨 <b>Tips for Colorful Markdown</b></summary>
  
  - Use <span style="color:orange;">**bold**</span> for commands.
  - Add emojis for quick visual guides.
  - Use code blocks for syntax highlighting.
  - Organize sections with headers and horizontal rules.
</details>