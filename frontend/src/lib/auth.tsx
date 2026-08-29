'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { api, setToken, clearToken, setUser, getUser } from './api';

interface AuthUser {
  id: number;
  name: string;
  email: string;
  role: string;
  org_id: number;
}

interface AuthCtx {
  user: AuthUser | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (data: any) => Promise<void>;
  logout: () => void;
}

const Ctx = createContext<AuthCtx>(null!);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUserState] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const existing = getUser();
    if (existing) {
      setUserState(existing);
    }
    setLoading(false);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const res = await api.login({ email, password });
    setToken(res.access_token);
    setUser(res.user);
    setUserState(res.user);
    router.push('/dashboard');
  }, [router]);

  const signup = useCallback(async (data: any) => {
    const res = await api.signup(data);
    setToken(res.access_token);
    setUser(res.user);
    setUserState(res.user);
    router.push('/dashboard');
  }, [router]);

  const logout = useCallback(() => {
    clearToken();
    setUserState(null);
    router.push('/');
  }, [router]);

  return (
    <Ctx.Provider value={{ user, loading, login, signup, logout }}>
      {children}
    </Ctx.Provider>
  );
}

export function useAuth() {
  return useContext(Ctx);
}
