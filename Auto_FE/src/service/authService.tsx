import { API_BASE } from '../api';
export const login = async (email: string, password: string) => {
  try {
    const response = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.message || 'Đăng nhập thất bại');
    }
    return data;
  } catch (error) {
    throw error;
  }
};