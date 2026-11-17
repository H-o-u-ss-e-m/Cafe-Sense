import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RoomStatus } from '../models/table.model';
import { MenuItem, Formula, Category, CafeInfo } from '../models/menu.model';

export interface VideoSourceOption {
  id: string;
  name: string;
  type: string;
  available: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private http = inject(HttpClient);
  public readonly baseUrl = 'http://localhost:5000';

  /** Récupère l'état temps réel des tables (caméra vision) */
  getRoomStatus(): Observable<RoomStatus> {
    return this.http.get<RoomStatus>(`${this.baseUrl}/api/status`);
  }

  /** Récupère la liste des sources vidéo disponibles (Webcam + Fichiers vidéos) */
  getVideoSources(): Observable<{ current_source: string; sources: VideoSourceOption[] }> {
    return this.http.get<{ current_source: string; sources: VideoSourceOption[] }>(`${this.baseUrl}/api/vision/sources`);
  }

  /** Bascule dynamiquement la source vidéo (Webcam '0', 'CAM1.mp4', 'CAM2.mp4', 'CAM3.mp4') */
  switchVideoSource(source: string): Observable<{ success: boolean; message: string; current_source: string }> {
    return this.http.post<{ success: boolean; message: string; current_source: string }>(`${this.baseUrl}/api/vision/source`, { source });
  }

  /** Force la recalibration de la détection des tables */
  recalibrate(): Observable<{ success: boolean; message: string }> {
    return this.http.post<{ success: boolean; message: string }>(`${this.baseUrl}/api/recalibrate`, {});
  }

  /** Récupère le catalogue complet (articles, catégories, formules, infos café) */
  getFullMenu(): Observable<{
    cafe: CafeInfo;
    categories: Category[];
    items: MenuItem[];
    formulas: Formula[];
  }> {
    return this.http.get<any>(`${this.baseUrl}/api/menu`);
  }

  /** Filtre le menu par catégorie, prix max ou tag */
  filterMenu(params: { category?: string; max_price?: number; tag?: string }): Observable<MenuItem[]> {
    return this.http.get<MenuItem[]>(`${this.baseUrl}/api/menu/filter`, { params: params as any });
  }

  /** Envoie un message au modèle IA conversationnel */
  sendChatMessage(message: string): Observable<{
    reply: string;
    intent: string;
    mode: string;
    items?: any[];
    tables_info?: any;
  }> {
    return this.http.post<any>(`${this.baseUrl}/api/chat`, { message });
  }

  /** URL du flux vidéo MJPEG en direct */
  getVideoStreamUrl(): string {
    return `${this.baseUrl}/video_feed`;
  }
}
