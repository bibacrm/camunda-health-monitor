# Contributing to Camunda Health Monitor

Thank you for your interest in contributing! This guide will help you get started.

## Quick Start

```bash
# Fork and clone
git clone https://github.com/bibacrm/camunda-health-monitor.git
cd camunda-health-monitor

# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Configure .env with your settings, then run
python app.py
```

## How to Contribute

### Reporting Issues

- Search existing issues first
- Provide clear reproduction steps
- Include versions (Camunda, PostgreSQL, Python)
- Add screenshots if applicable

### Submitting Pull Requests

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Make your changes
4. Test thoroughly
5. Commit: `git commit -m 'Add AmazingFeature'`
6. Push: `git push origin feature/AmazingFeature`
7. Open Pull Request

## Development Guidelines

### Code Style

- Follow PEP 8
- Maximum line length: 120 characters
- Add docstrings to functions
- Use type hints where helpful

**Example:**
```python
def collect_metrics(node_url: str) -> dict:
    """
    Collect metrics from Camunda node
    
    Args:
        node_url: Camunda REST API URL
        
    Returns:
        Dictionary with node metrics
    """
    # Implementation
```

### Project Structure

```
├── app.py              # Flask application factory
├── config.py           # Configuration management
├── wsgi.py             # Production WSGI entry
├── routes/             # HTTP endpoints (blueprints)
├── services/           # Business logic
├── helpers/            # Utilities
└── templates/          # HTML templates
```

### Testing Before PR

- [ ] Fresh install works
- [ ] Single and multi-node configurations
- [ ] Error scenarios (node down, DB unavailable)
- [ ] Dark mode works
- [ ] Responsive design (desktop, mobile)
- [ ] All endpoints return expected data

### Documentation

Update documentation when you:
- Add new features
- Change configuration options
- Modify API endpoints
- Fix unclear behavior

## Areas for Contribution

- 🐛 Bug fixes
- ✨ New features (discuss first via issue)
- 📝 Documentation improvements
- 🧪 Test coverage
- 🎨 UI/UX enhancements
- 🌍 Internationalization

## Questions?

- Open a GitHub Discussion for questions
- Create an Issue for bugs
- Check existing Issues/PRs first

## Recognition

Contributors are credited in:
- Project README
- Release notes
- Git commit history

---

**Every contribution helps!** Thank you for making this project better. 🎉

