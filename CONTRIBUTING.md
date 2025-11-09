# Contributing to Camunda Health Monitor

Thank you for your interest in contributing to Camunda Health Monitor! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with a clear title and description
3. **Include details:**
   - Your Camunda version
   - Your PostgreSQL version
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Screenshots (if applicable)

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

## 🔧 Development Setup

### Prerequisites

- Python 3.8 or higher
- PostgreSQL 12 or higher
- Git
- A Camunda 7.x instance for testing

### Local Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/camunda-health-monitor.git
cd camunda-health-monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Then run the application
python app.py
```

### Running with Docker

```bash
# Build the image
docker build -t camunda-health-monitor:dev .

# Run with docker-compose
docker-compose up
```

## 📝 Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Maximum line length: 120 characters

### Code Organization

```python
# Example function structure
def fetch_node_data(node_name, node_url):
    """
    Fetch health data from a single Camunda node
    
    Args:
        node_name: Name of the node
        node_url: REST API URL of the node
    
    Returns:
        dict: Node health metrics
    """
    # Implementation
    pass
```

### Frontend Code

- Use Alpine.js for interactivity
- Keep JavaScript functions small and focused
- Use Tailwind utility classes
- Ensure dark mode compatibility
- Test responsive design

## 🧪 Testing

### Manual Testing

Before submitting a PR, test:

1. **Fresh installation** - Does it work from scratch?
2. **Different configurations** - Single node, multiple nodes, with/without JMX
3. **Error scenarios** - Unavailable nodes, database errors
4. **Browser compatibility** - Chrome, Firefox, Safari
5. **Dark mode** - Both light and dark themes
6. **Responsive design** - Desktop, tablet, mobile

### Adding Tests

We welcome test contributions! Areas that need testing:

- Database query functions
- API endpoint responses
- Error handling
- Configuration parsing
- JMX metrics parsing

## 📚 Documentation

### When to Update Documentation

- Adding new features
- Changing configuration options
- Modifying API endpoints
- Fixing bugs that weren't clear in docs

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add screenshots for UI changes
- Keep README.md up to date

## 🎯 Feature Requests

### Proposing New Features

1. **Check existing issues** for similar proposals
2. **Open a discussion** issue explaining:
   - The problem it solves
   - Your proposed solution
   - Alternative approaches considered
   - Impact on existing functionality

## 🚀 Release Process

### Version Numbers

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in relevant files
- [ ] Docker image built and tested
- [ ] Release notes prepared

## 🏗️ Project Structure

```
camunda-health-monitor/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Dashboard UI
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container build
├── docker-compose.yml    # Docker orchestration
├── .env.example          # Configuration template
├── README.md             # Project documentation
├── CONTRIBUTING.md       # This file
└── LICENSE               # MIT License
```

## 💡 Tips for Contributors

### Good First Issues

Look for issues labeled `good-first-issue` - these are great starting points for new contributors.

### Communication

- Be respectful and constructive
- Ask questions if something is unclear
- Provide context in your comments
- Be patient - maintainers are often volunteers

### Code Review

- Address all review comments
- Don't take feedback personally
- Explain your reasoning if you disagree
- Keep discussions focused on the code

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create a GitHub Issue
- **Security**: Email security concerns privately (see README)

## 🙏 Recognition

Contributors will be:
- Listed in the project README
- Credited in release notes
- Acknowledged in commit history

---

**Thank you for contributing to Camunda Health Monitor!** 🎉

Every contribution, no matter how small, helps make this project better for everyone.