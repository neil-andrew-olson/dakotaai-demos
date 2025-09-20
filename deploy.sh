#!/bin/bash

echo "🚀 Deploying CIFAR-10 Image Classifier to GitHub Pages"
echo "===================================================="

# Check if we're in the right directory
if [ ! -d "apps/image-classifier" ]; then
    echo "❌ Error: apps/image-classifier directory not found"
    echo "Make sure you're in the project root directory"
    exit 1
fi

echo "📦 Building static export..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"
echo "📂 Static files ready in 'out/' directory"
echo ""
echo "🔧 Next Steps:"
echo "1. Go to your GitHub repository: https://github.com/neil-andrew-olson/dakotaai-demos"
echo "2. Go to Settings → Pages"
echo "3. Set Source to 'GitHub Actions'"
echo "4. Push this commit to trigger automatic deployment"
echo ""
echo "🌐 Your site will be available at:"
echo "https://neil-andrew-olson.github.io/dakotaai-demos/apps/image-classifier/"
echo ""
echo "🎉 Ready for deployment!"
