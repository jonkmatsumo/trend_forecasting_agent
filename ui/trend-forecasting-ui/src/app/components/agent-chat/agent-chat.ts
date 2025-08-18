import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { FormControl, Validators, ReactiveFormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { CommonModule } from '@angular/common';
import { AgentService } from '../../services/agent.service';
import { ChatMessage } from '../../models/agent.models';

@Component({
  selector: 'app-agent-chat',
  templateUrl: './agent-chat.html',
  styleUrls: ['./agent-chat.scss'],
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule
  ],
  standalone: true
})
export class AgentChatComponent implements OnInit {
  @ViewChild('chatContainer') chatContainer!: ElementRef;
  
  messages: ChatMessage[] = [];
  queryControl = new FormControl('', [Validators.required, Validators.maxLength(1000)]);
  isLoading = false;
  sessionId = this.generateSessionId();

  constructor(private agentService: AgentService) {}

  ngOnInit(): void {
    this.addSystemMessage('Hello! I\'m your trend forecasting assistant. Ask me anything about trends, forecasts, or model training.');
  }

  async sendMessage(): Promise<void> {
    if (this.queryControl.invalid || this.isLoading) return;

    const query = this.queryControl.value;
    if (!query) return;
    
    this.addUserMessage(query);
    this.queryControl.reset();
    this.isLoading = true;

    try {
      const response = await this.agentService.askAgent({
        message: query,
        context: this.sessionId
      }).toPromise();

      if (response) {
        this.addAgentMessage(response);
      }
    } catch (error) {
      this.addErrorMessage('Sorry, I encountered an error. Please try again.');
      console.error('Agent error:', error);
    } finally {
      this.isLoading = false;
      this.scrollToBottom();
    }
  }

  private addUserMessage(text: string): void {
    this.messages.push({
      id: Date.now(),
      type: 'user',
      text,
      timestamp: new Date()
    });
  }

  private addAgentMessage(response: any): void {
    this.messages.push({
      id: Date.now(),
      type: 'agent',
      text: response.text,
      data: response.data,
      metadata: response.metadata,
      timestamp: new Date()
    });
  }

  private addSystemMessage(text: string): void {
    this.messages.push({
      id: Date.now(),
      type: 'system',
      text,
      timestamp: new Date()
    });
  }

  private addErrorMessage(text: string): void {
    this.messages.push({
      id: Date.now(),
      type: 'error',
      text,
      timestamp: new Date()
    });
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      this.chatContainer.nativeElement.scrollTop = this.chatContainer.nativeElement.scrollHeight;
    }, 100);
  }

  handleKeyDown(event: Event): void {
    const keyboardEvent = event as KeyboardEvent;
    if (keyboardEvent.key === 'Enter' && !keyboardEvent.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  private generateSessionId(): string {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }
}
