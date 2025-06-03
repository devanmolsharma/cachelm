#!/bin/bash

set -e  # Exit on error

# Clear dist/
echo "🧹 Clearing dist/ directory..."
rm -rf dist/
mkdir -p dist

# Get current version from pyproject.toml
current_version=$(grep '^version =' pyproject.toml | cut -d '"' -f2)
IFS='.' read -r major minor patch <<< "$current_version"

echo "📦 Current version: $current_version"
echo "Choose version bump:"
select choice in "Patch" "Minor" "Major"; do
    case $choice in
        Patch )
            patch=$((patch + 1))
            break
            ;;
        Minor )
            minor=$((minor + 1))
            patch=0
            break
            ;;
        Major )
            major=$((major + 1))
            minor=0
            patch=0
            break
            ;;
        * )
            echo "Invalid choice. Try again."
            ;;
    esac
done

new_version="$major.$minor.$patch"
echo "🔁 Updating version to $new_version..."

# Update pyproject.toml
sed -i.bak "s/version = \".*\"/version = \"$new_version\"/" pyproject.toml
rm pyproject.toml.bak

# Build project
echo "🔨 Building project..."
uv pip install --upgrade build
uv run python3 -m build


# Update requirements.txt
echo "📄 Updating requirements.txt..."
uv pip freeze > requirements.txt

# Deploy (customize as needed)
echo "🚀 Deploying..."
uv pip install --upgrade twine
uv run python3 -m twine upload --repository pypi --username __token__ dist/* --verbose



echo "✅ Done! Deployed version $new_version"
