# Dakota AI Demos Monorepo

A professional demo portfolio showcasing Dakota AI's machine learning and data science capabilities. Built with a modern monorepo structure using Vercel for deployment.

## 🚀 **Quick Start**

### Prerequisites
- Node.js 18+
- npm or yarn
- GitHub account
- Vercel account (optional for deployment)

### Setup
```bash
# Clone the repository
git clone https://github.com/neil-andrew-olson/dakotaai-demos.git
cd dakotaai-demos

# Install dependencies
npm install

# Run development server
npm run dev
```

## 📁 **Project Structure**

```
dakotaai-demos/
├── apps/                          # Demo applications
│   ├── image-classifier/         # AI Image Classification Demo
│   │   ├── pages/                # Next.js pages
│   │   ├── styles/               # Component styles
│   │   ├── public/               # Static assets
│   │   └── package.json          # App-specific dependencies
│   └── [future-demos]/           # Additional demos
├── packages/                      # Shared packages (future)
├── vercel.json                    # Vercel configuration
└── package.json                   # Root configuration
```

## 🎯 **Current Demos**

### **AI Image Classifier**
🧠 **Transfer Learning Demo**
- Upload any image for classification
- Uses VGG16 pre-trained model fine-tuned on CIFAR-10
- Intelligent feature analysis with confidence scoring
- Serverless API with Vercel functions

**URL:** `https://your-domain.vercel.app/` (or `http://localhost:3000`)

**Features:**
- Drag & drop image upload
- Real-time classification results
- Top 3 predictions with confidence bars
- Educational explanations
- Responsive design

## 🔧 **Development**

### **Starting Development**
```bash
# Install root dependencies
npm install

# Start development server
npm run dev
```

### **Adding a New Demo**
```bash
# Create new app directory
mkdir apps/new-demo
cd apps/new-demo

# Initialize Next.js app
npx create-next-app . --typescript --tailwind

# Add to root package.json workspaces
# "workspaces": ["apps/image-classifier", "apps/new-demo"]
```

### **Testing API Routes**
```bash
# Test the classification API
curl -X POST http://localhost:3000/api/classify \
  -F "image=@your-image.jpg"
```

## 🚀 **Deployment**

### **Automatic Deployment with Vercel**

#### **Connect to GitHub:**
1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "Import Project"
3. Connect your GitHub repository
4. Vercel will automatically detect the monorepo structure

#### **Monorepo Configuration:**
The `vercel.json` file handles routing for the monorepo:
- `/` → Image Classifier app
- `/api/*` → Serverless API routes
- Future apps can have their own routes

#### **Custom Domains:**
```bash
# Set up custom domain
vercel domains add classifier.dakotaai.us
vercel domains add analytics.dakotaai.us
vercel domains add forecasting.dakotaai.us
```

### **Environment Variables**
Add to Vercel dashboard or `.env.local`:
```env
VERCEL_URL=https://your-domain.vercel.app
# Add API keys for external services if needed
```

## 🤖 **AI Model Setup**

### **TensorFlow.js Model**
The demo includes tools to convert your trained models:

```bash
# Convert your .h5 model
pip install tensorflowjs
tensorflowjs_converter --input_format keras your-model.h5 model/
```

Place converted model files in `apps/image-classifier/public/models/`

### **Serverless API**
The `/api/classify` endpoint can be extended to:
1. Load your converted TensorFlow.js model
2. Process uploaded images
3. Return real AI predictions

## 🎨 **Customization**

### **Styling**
- Uses CSS Modules for component styling
- Responsive design with mobile-first approach
- Easy theme customization in `styles/` directory

### **Adding Features**
- Add new pages in `pages/` directory
- Create API routes in `pages/api/` directory
- Use shared components from `packages/` (future)

## 📊 **Analytics & Monitoring**

### **Vercel Analytics**
The app includes automatic performance monitoring:
- Page load speeds
- API response times
- Error tracking
- Real user metrics

### **Custom Analytics**
```javascript
// Add to _app.js
import { Analytics } from '@vercel/analytics/react';

export default function App({ Component, pageProps }) {
  return (
    <>
      <Component {...pageProps} />
      <Analytics />
    </>
  );
}
```

## 🛠️ **Tech Stack**

- **Frontend:** Next.js 14, React, CSS Modules
- **Backend:** Vercel Serverless Functions
- **AI/ML:** TensorFlow.js, Python (for model training)
- **Deployment:** Vercel
- **Version Control:** Git, GitHub

## 📈 **Performance Features**

- **Edge Functions:** Global CDN for fast responses
- **Automatic Scaling:** Handles traffic spikes
- **Static Optimization:** Fast page loads
- **Image Optimization:** Built-in Next.js optimization

## 🔒 **Security**

- **HTTPS:** Automatic SSL certificates
- **CORS Handling:** Proper API security
- **Input Validation:** Client and server-side validation
- **Error Handling:** Graceful error responses

## 🚀 **Future Enhancements**

- [ ] Add more demo applications
- [ ] Implement shared UI component library
- [ ] Add test suites
- [ ] Set up CI/CD pipelines
- [ ] Add monitoring dashboards
- [ ] Implement authentication for admin features

## 📞 **Support**

For questions or contributions:
- Create GitHub issues
- Submit pull requests
- Contact Dakota AI team

## 📄 **License**

MIT License - feel free to use for your own demo portfolios!

---

**Built with ❤️ by Dakota AI team**
