export interface ChatMessage {
  sender: 'user' | 'agent';
  text: string;
  time: string;
  intent?: string;
  mode?: string;
  items?: any[];
}
