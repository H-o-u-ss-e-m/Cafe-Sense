import { Component, OnInit, OnDestroy, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, VideoSourceOption } from './core/services/api.service';
import { RoomStatus, Table } from './core/models/table.model';
import { MenuItem, Formula, Category, CafeInfo } from './core/models/menu.model';
import { ChatMessage } from './core/models/chat.model';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit, OnDestroy {
  private api = inject(ApiService);

  // Onglet actif : 'dashboard' | 'menu' | 'formulas' | 'info'
  activeTab = signal<'dashboard' | 'menu' | 'formulas' | 'info'>('dashboard');

  // État de la salle & caméras
  roomStatus = signal<RoomStatus>({
    total_tables: 0,
    free_count: 0,
    occupied_count: 0,
    leaving_count: 0,
    tables: []
  });
  videoStreamUrl = signal<string>('');
  isStreamActive = signal<boolean>(true);
  statusPollingTimer: any = null;

  // Gestion dynamique des Sources Vidéo (Webcam vs Fichiers vidéo)
  availableSources = signal<VideoSourceOption[]>([]);
  currentSource = signal<string>('0');
  isSwitchingSource = signal<boolean>(false);

  // Données Menu
  cafeInfo = signal<CafeInfo | null>(null);
  categories = signal<Category[]>([]);
  allMenuItems = signal<MenuItem[]>([]);
  formulas = signal<Formula[]>([]);

  // Filtres Menu
  selectedCategory = signal<string>('all');
  searchQuery = signal<string>('');
  filterGlutenFree = signal<boolean>(false);
  filterVegetarian = signal<boolean>(false);
  filterMaxPrice = signal<number>(50);

  // Chatbot IA
  isChatOpen = signal<boolean>(false);
  chatInput = signal<string>('');
  isChatLoading = signal<boolean>(false);
  chatMessages = signal<ChatMessage[]>([
    {
      sender: 'agent',
      text: '👋 Bonjour et marhba bik au Coffee House Lac ! Je suis votre agent IA connecté en direct à notre menu et à nos caméras. Que puis-je vous servir ou vous renseigner aujourd\'hui ?',
      time: this.getCurrentTime()
    }
  ]);

  quickSuggestions = [
    'Combien coûte le capucin ?',
    '9adeh soum l\'express ?',
    'Fama blasa tawwa ?',
    'Plats sans gluten ?',
    'Avez-vous une formule pour 2 ?',
    'C\'est quoi le code wifi ?'
  ];

  ngOnInit() {
    this.updateStreamUrl();
    this.fetchRoomStatus();
    this.fetchVideoSources();
    this.fetchMenuData();

    // Rafraîchissement automatique de la détection vidéo toutes les secondes
    this.statusPollingTimer = setInterval(() => {
      this.fetchRoomStatus();
    }, 1000);
  }

  ngOnDestroy() {
    if (this.statusPollingTimer) {
      clearInterval(this.statusPollingTimer);
    }
  }

  getCurrentTime(): string {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  updateStreamUrl() {
    this.videoStreamUrl.set(`${this.api.getVideoStreamUrl()}?t=${Date.now()}`);
  }

  fetchVideoSources() {
    this.api.getVideoSources().subscribe({
      next: (res) => {
        this.availableSources.set(res.sources || []);
        this.currentSource.set(res.current_source || '0');
      },
      error: (err) => console.error('Erreur récupération sources vidéo:', err)
    });
  }

  changeVideoSource(newSource: string) {
    if (this.isSwitchingSource() || this.currentSource() === newSource) return;
    this.isSwitchingSource.set(true);
    this.currentSource.set(newSource);

    // Vider temporairement l'URL pour couper l'ancien flux et éviter les saccades/gels d'écran
    this.videoStreamUrl.set('');

    this.api.switchVideoSource(newSource).subscribe({
      next: () => {
        // Laisser 1.2s au backend pour initialiser proprement la caméra DirectShow ou la vidéo
        setTimeout(() => {
          this.updateStreamUrl();
          this.fetchRoomStatus();
          // Masquer le spinner dès que la nouvelle source commence à émettre
          setTimeout(() => {
            this.isSwitchingSource.set(false);
          }, 800);
        }, 1200);
      },
      error: (err) => {
        console.error('Erreur changement source vidéo:', err);
        this.updateStreamUrl();
        this.isSwitchingSource.set(false);
      }
    });
  }

  getSourceLabel(sourceId: string): string {
    const found = this.availableSources().find(s => s.id === sourceId);
    if (found) return found.name;
    if (sourceId === '0') return 'Caméra PC (Webcam en Direct)';
    if (sourceId.toLowerCase().includes('cam1')) return '🎥 Caméra 1 - Salle Principale';
    if (sourceId.toLowerCase().includes('cam2')) return '🎥 Caméra 2 - Terrasse';
    if (sourceId.toLowerCase().includes('cam3')) return '🎥 Caméra 3 - Comptoir & Entrée';
    return sourceId;
  }

  fetchRoomStatus() {
    this.api.getRoomStatus().subscribe({
      next: (status) => {
        this.roomStatus.set(status);
        this.isStreamActive.set(true);
      },
      error: () => {
        this.isStreamActive.set(false);
      }
    });
  }

  fetchMenuData() {
    this.api.getFullMenu().subscribe({
      next: (data) => {
        this.cafeInfo.set(data.cafe);
        this.categories.set(data.categories || []);
        this.allMenuItems.set(data.items || []);
        this.formulas.set(data.formulas || []);
      },
      error: (err) => console.error('Erreur chargement menu:', err)
    });
  }

  recalibrateTables() {
    if (!confirm('Voulez-vous réinitialiser la calibration automatique des tables ?')) return;
    this.api.recalibrate().subscribe({
      next: (res) => alert(res.message || 'Calibration relancée !'),
      error: () => alert('Erreur lors de la recalibration.')
    });
  }

  // Filtrage dynamique des articles du menu
  get filteredItems(): MenuItem[] {
    return this.allMenuItems().filter((item) => {
      // Catégorie
      if (this.selectedCategory() !== 'all' && item.category !== this.selectedCategory()) {
        return false;
      }
      // Recherche texte
      if (this.searchQuery().trim()) {
        const q = this.searchQuery().toLowerCase();
        const inName = item.name.toLowerCase().includes(q);
        const inDesc = item.description.toLowerCase().includes(q);
        if (!inName && !inDesc) return false;
      }
      // Sans gluten
      if (this.filterGlutenFree() && !item.tags.includes('sans_gluten')) {
        return false;
      }
      // Végétarien
      if (this.filterVegetarian() && !item.tags.includes('vegetarien')) {
        return false;
      }
      // Prix max
      if (item.price > this.filterMaxPrice()) {
        return false;
      }
      return true;
    });
  }

  selectCategory(catId: string) {
    this.selectedCategory.set(catId);
  }

  // Chatbot Actions
  toggleChat() {
    this.isChatOpen.update((v) => !v);
  }

  sendQuickQuestion(q: string) {
    this.chatInput.set(q);
    this.sendMessage();
  }

  sendMessage() {
    const text = this.chatInput().trim();
    if (!text || this.isChatLoading()) return;

    // Ajouter message client
    const userMsg: ChatMessage = {
      sender: 'user',
      text: text,
      time: this.getCurrentTime()
    };
    this.chatMessages.update((msgs) => [...msgs, userMsg]);
    this.chatInput.set('');
    this.isChatLoading.set(true);

    this.api.sendChatMessage(text).subscribe({
      next: (resp) => {
        this.isChatLoading.set(false);
        const agentMsg: ChatMessage = {
          sender: 'agent',
          text: resp.reply,
          time: this.getCurrentTime(),
          intent: resp.intent,
          mode: resp.mode,
          items: resp.items
        };
        this.chatMessages.update((msgs) => [...msgs, agentMsg]);
        this.scrollChatToBottom();
      },
      error: () => {
        this.isChatLoading.set(false);
        const errorMsg: ChatMessage = {
          sender: 'agent',
          text: 'Désolé, impossible de communiquer avec le serveur pour le moment.',
          time: this.getCurrentTime()
        };
        this.chatMessages.update((msgs) => [...msgs, errorMsg]);
        this.scrollChatToBottom();
      }
    });
  }

  scrollChatToBottom() {
    setTimeout(() => {
      const container = document.getElementById('chat-scroll-container');
      if (container) container.scrollTop = container.scrollHeight;
    }, 100);
  }
}
