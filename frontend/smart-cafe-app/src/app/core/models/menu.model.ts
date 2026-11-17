export interface MenuItem {
  id: number;
  name: string;
  category: string;
  price: number;
  description: string;
  ingredients: string[];
  allergens: string[];
  tags: string[];
  calories: number;
  available: boolean;
  image?: string;
  aliases?: string[];
  availability_note?: string;
}

export interface Formula {
  id: number;
  name: string;
  price: number;
  description: string;
  includes: string[];
  conditions: string;
  persons: number;
  available?: boolean;
}

export interface Category {
  id: string;
  name: string;
  aliases: string[];
}

export interface CafeInfo {
  name: string;
  tagline?: string;
  address: string;
  phone: string;
  whatsapp: string;
  wifi: { available: boolean; free: boolean; note: string };
  hours_groups: { days_fr: string; open: string; close: string }[];
  delivery: { available: boolean; fee: number; min_order: number; hours: string; zones: string[] };
  services: { parking: string; halal: string; terrace: boolean };
}
