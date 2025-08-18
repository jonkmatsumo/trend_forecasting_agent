# Trend Forecasting UI

A modern Angular-based user interface for the Trend Forecasting Agent system, providing an interactive chat interface and comprehensive API testing capabilities.

## 🚀 Features

### 🤖 Agent Chat Interface
- **Interactive Chat**: Real-time conversation with the trend forecasting AI agent
- **Session Management**: Persistent chat sessions with unique session IDs
- **Message Types**: Support for user, agent, system, and error messages
- **Auto-scroll**: Automatic scrolling to latest messages
- **Input Validation**: Form validation with character limits and required fields
- **Loading States**: Visual feedback during agent processing

### 🔧 API Testing Suite
- **Endpoint Testing**: Pre-configured endpoints for all backend APIs
- **Request Builder**: Dynamic form-based request construction
- **Response Viewer**: Formatted JSON response display with syntax highlighting
- **Integration Testing**: Automated testing of all backend endpoints
- **CORS Testing**: Cross-origin resource sharing configuration validation
- **Proxy Testing**: Proxy configuration validation
- **Copy to Clipboard**: Easy response copying functionality

### 🎨 Styling Demo
- **Notification System**: Comprehensive notification service with multiple types
- **Glass Morphism**: Modern UI effects with glass morphism styling
- **Action Buttons**: Interactive notifications with custom actions
- **Duration Control**: Configurable notification display times
- **Dismissible Notifications**: User-controlled notification dismissal

### 🛠️ Shared Components
- **Notification System**: Toast-style notifications with multiple types (success, error, warning, info)
- **Loading Spinner**: Reusable loading indicators
- **Header Component**: Application header with navigation
- **Sidebar**: Navigation sidebar component
- **Layout Components**: Responsive layout management

## 🏗️ Architecture

### Core Services
- **AgentService**: Handles communication with the AI agent backend
- **ApiService**: Manages HTTP requests to the backend API
- **ApiTestService**: Provides comprehensive API testing functionality
- **NotificationService**: Manages application-wide notifications
- **ConfigService**: Centralized configuration management

### Data Models
- **Agent Models**: Chat messages, agent requests/responses
- **API Models**: Endpoint definitions, request/response structures
- **Shared Models**: Common data structures

### Component Structure
```
src/app/
├── components/
│   ├── agent-chat/          # Main chat interface
│   ├── api-tester/          # API testing interface
│   └── shared/              # Reusable components
│       ├── notification/    # Notification system
│       ├── header/          # Application header
│       ├── sidebar/         # Navigation sidebar
│       └── loading-spinner/ # Loading indicators
├── pages/
│   ├── agent-chat-page/     # Chat page wrapper
│   ├── api-tester-page/     # API testing page wrapper
│   └── styling-demo-page/   # UI demonstration page
├── services/                # Business logic services
└── models/                  # TypeScript interfaces
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v18 or higher)
- Angular CLI (v20.1.6)
- Backend API running on `http://localhost:5000`

### Installation
```bash
# Install dependencies
npm install

# Start development server
npm start
```

The application will be available at `http://localhost:4200`

### Available Scripts
```bash
npm start          # Start development server
npm run build      # Build for production
npm run test       # Run unit tests
npm run watch      # Build with watch mode
```

## 🔧 Configuration

### Environment Variables
The application uses environment-based configuration:

- **Development**: `src/environments/environment.ts`
- **Production**: `src/environments/environment.prod.ts`

Key configuration options:
- `apiUrl`: Backend API base URL (default: `http://localhost:5000`)
- `agentUrl`: AI agent endpoint (default: `http://localhost:5000/agent`)
- `enableLogging`: Enable/disable debug logging

### API Endpoints
The UI is pre-configured to work with these backend endpoints:

- **Health Check**: `GET /health`
- **Get Trends**: `POST /trends`
- **Train Model**: `POST /models/train`
- **Model Prediction**: `POST /models/{model_id}/predict`
- **Get Model**: `GET /models/{model_id}`
- **List Models**: `GET /models`
- **Clear Cache**: `POST /trends/cache/clear`
- **Cache Stats**: `GET /trends/cache/stats`
- **Trends Summary**: `POST /trends/summary`
- **Compare Trends**: `POST /trends/compare`

## 🎨 Styling

### Design System
- **Angular Material**: Material Design 3 components
- **Custom SCSS**: Extended styling with variables and mixins
- **Responsive Design**: Mobile-first responsive layout
- **Glass Morphism**: Modern UI effects
- **Color Palette**: Azure and blue-based color scheme

### Key Styling Features
- Custom CSS variables for consistent theming
- Responsive container classes
- Typography system with Roboto font
- Smooth animations and transitions
- Accessibility-focused design

## 🧪 Testing

### Unit Tests
```bash
npm run test
```

### Test Coverage
- Component testing with Jasmine/Karma
- Service testing with dependency injection
- Model validation testing
- API integration testing

## 📱 Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## 🔗 Dependencies

### Core Dependencies
- **Angular**: v20.1.0 (Core framework)
- **Angular Material**: v20.1.6 (UI components)
- **RxJS**: v7.8.0 (Reactive programming)
- **Chart.js**: v4.5.0 (Data visualization)
- **Moment.js**: v2.30.1 (Date handling)

### Development Dependencies
- **TypeScript**: v5.8.2
- **Angular CLI**: v20.1.6
- **Karma**: v6.4.0 (Testing framework)
- **Jasmine**: v5.8.0 (Testing library)

## 🤝 Contributing

1. Follow Angular coding standards
2. Write unit tests for new features
3. Update documentation for API changes
4. Ensure responsive design compatibility
5. Test across different browsers

## 📄 License

This project is part of the Trend Forecasting Agent system.
