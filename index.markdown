---
layout: home
title: Home
---

## 👋 Introduction
Welcome! I am [Your Name], a researcher and developer focused on [Topic].

---

## 🧑‍💻 About Me
I enjoy building tools that solve [Problem]. Currently, I am exploring [Research Area].

---

## 🛠️ Skills
* **Languages:** Python, C++, SQL, Ruby
* **Research:** LaTeX, Data Visualization, Signal Processing
* **Tools:** Git, Docker, PyTorch

---

## 🚀 Projects
{% for project in site.projects %}
### [{{ project.title }}]({{ project.url }})
{{ project.description }}
{% endfor %}

---

## 📓 Research Journal
{% for post in site.posts %}
* **{{ post.date | date: "%b %d, %Y" }}** - [{{ post.title }}]({{ post.url }})
{% endfor %}