# Trend Forecasting Agent UI

A modern, enterprise-grade Angular 20 application providing an interactive interface for the Trend Forecasting Agent system. Built with TypeScript, Angular Material, and SCSS, this UI offers both a conversational AI interface and comprehensive API testing capabilities.

## 🚀 Features

### 🤖 Agent Chat Interface
- **Real-time Conversation**: Interactive chat with the trend forecasting AI agent
- **Session Management**: Persistent chat sessions with unique session IDs
- **Message Types**: Support for user, agent, system, and error messages with visual differentiation
- **Auto-scroll**: Automatic scrolling to latest messages with smooth animations
- **Input Validation**: Form validation with character limits (1000 chars) and required fields
- **Loading States**: Visual feedback during agent processing with progress indicators
- **Error Handling**: Graceful error handling with user-friendly error messages
- **Context Preservation**: Maintains conversation context across messages

### 🔧 API Testing Suite
- **Comprehensive Endpoint Testing**: Pre-configured endpoints for all backend APIs
- **Dynamic Request Builder**: Form-based request construction with JSON validation
- **Response Viewer**: Formatted JSON response display with syntax highlighting
- **Integration Testing**: Automated testing of all backend endpoints with detailed results
- **CORS Testing**: Cross-origin resource sharing configuration validation
- **Proxy Testing**: Proxy configuration validation and testing
- **Copy to Clipboard**: One-click response copying functionality
- **Request History**: Track and replay previous API calls
- **Performance Metrics**: Response time tracking and performance analysis

### 🎨 Styling & UI Components
- **Modern Design System**: Material Design 3 with custom Azure/Blue color palette
- **Glass Morphism Effects**: Contemporary UI effects with backdrop blur
- **Responsive Layout**: Mobile-first responsive design with breakpoint optimization
- **Notification System**: Toast-style notifications with multiple types (success, error, warning, info)
- **Loading Indicators**: Skeleton loaders and spinners for better UX
- **Form Components**: Reusable form elements with validation
- **Card Layouts**: Consistent card-based content presentation

### 🛠️ Shared Components
- **Notification System**: Comprehensive notification service with configurable duration and actions
- **Loading Spinner**: Reusable loading indicators with progress tracking
- **Header Component**: Application header with navigation and branding
- **Sidebar**: Collapsible navigation sidebar with route highlighting
- **Layout Components**: Responsive layout management with grid systems
- **Skeleton Loader**: Content placeholders for loading states

### 🔍 Error Handling Demo
- **Form Validation**: Comprehensive form validation with custom validators
- **HTTP Error Simulation**: Test various HTTP error scenarios
- **Loading State Management**: Advanced loading state handling
- **Error Recovery**: Graceful error recovery mechanisms
- **Validation Examples**: Real-world validation scenarios

## 🏗️ Architecture

### Core Services
- **AgentService**: Handles communication with the AI agent backend with retry logic
- **ApiService**: Manages HTTP requests to the backend API with error handling
- **ApiTestService**: Provides comprehensive API testing functionality
- **NotificationService**: Manages application-wide notifications with queuing
- **ConfigService**: Centralized configuration management with environment support
- **ErrorHandlerService**: Centralized error handling with custom error types
- **ValidationService**: Form and data validation with custom validators
- **LoadingService**: Global loading state management

### Data Models
- **Agent Models**: Chat messages, agent requests/responses, configuration
- **API Models**: Endpoint definitions, request/response structures, test results
- **Shared Models**: Common data structures and interfaces

### Component Structure
```
src/app/
├── components/
│   ├── agent-chat/          # Main chat interface with real-time messaging
│   ├── api-tester/          # Comprehensive API testing interface
│   └── shared/              # Reusable components
│       ├── notification/    # Toast notification system
│       ├── notification-container/ # Notification management
│       ├── header/          # Application header
│       ├── sidebar/         # Navigation sidebar
│       ├── loading-spinner/ # Loading indicators
│       └── skeleton-loader/ # Content placeholders
├── pages/
│   ├── agent-chat-page/     # Chat page wrapper
│   ├── api-tester-page/     # API testing page wrapper
│   ├── styling-demo-page/   # UI demonstration page
│   └── error-handling-demo-page/ # Error handling showcase
├── services/                # Business logic and API services
└── models/                  # TypeScript interfaces and types
```

### Routing Structure
- **Default Route**: `/` → `/agent-chat`
- **Agent Chat**: `/agent-chat` - Main conversation interface
- **API Testing**: `/api-tester` - Comprehensive API testing suite
- **Styling Demo**: `/styling-demo` - UI component showcase
- **Error Handling Demo**: `/error-handling-demo` - Error scenarios demonstration

## 🚀 Getting Started

### Prerequisites
- **Node.js**: v18 or higher
- **Angular CLI**: v20.1.6
- **Backend API**: Running on `http://localhost:5000`

### Installation
```bash
# Navigate to the UI directory
cd ui/trend-forecasting-ui

# Install dependencies
npm install

# Start development server
npm start
```

The application will be available at `http://localhost:4200`

### Available Scripts
```bash
npm start          # Start development server with proxy
npm run build      # Build for production
npm run test       # Run unit tests with coverage
npm run watch      # Build with watch mode
npm run lint       # Run ESLint
npm run e2e        # Run end-to-end tests (if configured)
```

## 🔧 Configuration

### Environment Configuration
The application uses environment-based configuration:

- **Development**: `src/environments/environment.ts`
- **Production**: `src/environments/environment.prod.ts`

#### Key Configuration Options
```typescript
{
  apiUrl: 'http://localhost:5000',           // Backend API base URL
  agentUrl: 'http://localhost:5000/agent',   // AI agent endpoint
  enableLogging: true,                       // Debug logging
  apiTimeout: 30000,                         // API request timeout (ms)
  apiRetryAttempts: 3,                       // Retry attempts for failed requests
  features: {
    notifications: { enabled: true },        // Notification system
    analytics: { enabled: false },           // Analytics tracking
    logging: { enabled: true }               // Application logging
  },
  ui: {
    theme: 'light',                          // UI theme
    language: 'en',                          // Language
    timezone: 'UTC'                          // Timezone
  },
  security: {
    cors: { enabled: true },                 // CORS configuration
    auth: { enabled: false },                // Authentication
    rateLimiting: { enabled: true }          // Rate limiting
  }
}
```

### Proxy Configuration
Development proxy configuration (`proxy.conf.json`):
```json
{
  "/api": {
    "target": "http://localhost:5000",
    "secure": false,
    "changeOrigin": true
  },
  "/agent": {
    "target": "http://localhost:5000",
    "secure": false,
    "changeOrigin": true
  }
}
```

### API Endpoints
The UI is pre-configured to work with these backend endpoints:

#### Core Endpoints
- **Health Check**: `GET /health`
- **Agent Chat**: `POST /agent/chat`
- **Agent Ask**: `POST /agent/ask`
- **Agent Health**: `GET /agent/health`
- **Agent Capabilities**: `GET /agent/capabilities`
- **Agent Config**: `GET /agent/config`

#### Trends API
- **Get Trends**: `POST /trends`
- **Trends Summary**: `POST /trends/summary`
- **Compare Trends**: `POST /trends/compare`
- **Clear Cache**: `POST /trends/cache/clear`
- **Cache Stats**: `GET /trends/cache/stats`

#### Models API
- **Train Model**: `POST /models/train`
- **Model Prediction**: `POST /models/{model_id}/predict`
- **Get Model**: `GET /models/{model_id}`
- **List Models**: `GET /models`

## 🎨 Styling & Design System

### Design Principles
- **Material Design 3**: Google's latest design system
- **Responsive Design**: Mobile-first approach with breakpoint optimization
- **Accessibility**: WCAG 2.1 AA compliance
- **Performance**: Optimized animations and transitions

### Color Palette
```scss
// Primary Colors
$primary-color: #3f51b5;
$secondary-color: #ff4081;
$success-color: #4caf50;
$error-color: #f44336;
$warning-color: #ff9800;
$info-color: #2196f3;

// Background Colors
$background-color: #fafafa;
$surface-color: #ffffff;
$card-background: #ffffff;
```

### Typography
- **Primary Font**: Roboto (Material Design)
- **Secondary Font**: Inter (for enhanced readability)
- **Display Font**: Poppins (for headings)
- **Monospace**: Courier New (for code)

### Component Styling
- **Glass Morphism**: Modern backdrop blur effects
- **Elevation System**: Consistent shadow hierarchy
- **Border Radius**: Progressive border radius scale
- **Spacing System**: 8px base unit spacing scale

## 🧪 Testing

### Testing Strategy
The application follows the testing pyramid approach:
- **Unit Tests**: 80%+ coverage target
- **Integration Tests**: Component and service integration
- **E2E Tests**: Complete user workflows

### Unit Testing
```bash
# Run all unit tests
npm run test

# Run tests with coverage
npm run test -- --coverage

# Run tests in watch mode
npm run test -- --watch
```

### Test Coverage
- **Components**: User interactions, data binding, lifecycle hooks
- **Services**: Business logic, API calls, error handling
- **Models**: Type validation and data transformations
- **Pipes**: Data transformation logic

### Testing Tools
- **Framework**: Jasmine + Karma
- **Mocking**: Angular TestBed, jasmine spies
- **HTTP Testing**: HttpClientTestingModule
- **Component Testing**: ComponentFixture, DebugElement

## 📱 Browser Support

### Supported Browsers
- **Chrome**: Latest 2 versions
- **Firefox**: Latest 2 versions
- **Safari**: Latest 2 versions
- **Edge**: Latest 2 versions

### Mobile Support
- **iOS Safari**: 12+
- **Chrome Mobile**: Latest
- **Samsung Internet**: Latest

## 🔗 Dependencies

### Core Dependencies
```json
{
  "@angular/animations": "^20.1.7",
  "@angular/cdk": "^20.1.6",
  "@angular/common": "^20.1.0",
  "@angular/compiler": "^20.1.0",
  "@angular/core": "^20.1.0",
  "@angular/forms": "^20.1.0",
  "@angular/material": "^20.1.6",
  "@angular/platform-browser": "^20.1.0",
  "@angular/platform-server": "^20.1.0",
  "@angular/router": "^20.1.0",
  "@angular/ssr": "^20.1.6"
}
```

### Development Dependencies
```json
{
  "@angular/build": "^20.1.6",
  "@angular/cli": "^20.1.6",
  "@angular/compiler-cli": "^20.1.0",
  "typescript": "^5.8.2",
  "karma": "^6.4.0",
  "jasmine": "^5.8.0"
}
```

## 🚀 Performance Features

### Optimization Strategies
- **Lazy Loading**: Route-based code splitting
- **Tree Shaking**: Unused code elimination
- **Bundle Optimization**: Webpack optimization
- **Caching**: HTTP caching strategies
- **Compression**: Gzip compression support

### Monitoring
- **Performance Metrics**: Response time tracking
- **Error Tracking**: Centralized error logging
- **User Analytics**: Usage pattern analysis
- **Health Checks**: Application health monitoring

## 🔒 Security Features

### Security Measures
- **CORS Configuration**: Proper cross-origin handling
- **Input Validation**: Comprehensive input sanitization
- **XSS Protection**: Content Security Policy
- **CSRF Protection**: Cross-site request forgery prevention
- **Rate Limiting**: API rate limiting support

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Follow Angular style guide
2. **TypeScript**: Use strict mode and proper typing
3. **Testing**: Write unit tests for new features
4. **Documentation**: Update documentation for API changes
5. **Accessibility**: Ensure WCAG compliance
6. **Performance**: Optimize for performance
7. **Security**: Follow security best practices

### Pull Request Process
1. Create feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Run linting and tests
5. Submit pull request with description

## 📄 License

This project is part of the Trend Forecasting Agent system and follows the same licensing terms.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information
4. Contact the development team

---

**Built with ❤️ using Angular 20, TypeScript, and Angular Material**
