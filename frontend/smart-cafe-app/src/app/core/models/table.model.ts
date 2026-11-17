export interface Table {
  table_id: number;
  status: 'LIBRE' | 'OCCUPEE' | 'EN DÉPART';
  persons_count: number;
  timer_display: string;
  leaving_countdown_sec?: number;
  bbox?: number[];
}

export interface RoomStatus {
  total_tables: number;
  free_count: number;
  occupied_count: number;
  leaving_count: number;
  tables: Table[];
}
