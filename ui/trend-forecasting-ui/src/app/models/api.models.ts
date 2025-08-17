export interface ApiEndpoint {
  name: string;
  method: string;
  path: string;
  description: string;
  bodyTemplate?: any;
}

export interface ApiRequest {
  url: string;
  method: string;
  headers: { [key: string]: string };
  body?: any;
}

export interface ApiResponse {
  status: number;
  statusText: string;
  data: any;
  headers: any;
  responseTime: number;
} 