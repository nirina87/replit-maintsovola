import os
import psycopg2
import bcrypt
import streamlit as st
from typing import Optional, Dict, Any

class AuthManager:
    """
    Gestionnaire d'authentification pour Maintso Vola
    """
    
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
    
    def get_connection(self):
        """Établit une connexion à la base de données"""
        return psycopg2.connect(self.db_url)
    
    def hash_password(self, password: str) -> str:
        """Hache un mot de passe"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Vérifie un mot de passe"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def register_user(self, nom: str, prenoms: str, email: str, telephone: str, 
                     password: str, investir: bool, cherche_investisseurs: bool) -> Dict[str, Any]:
        """
        Inscrit un nouvel utilisateur
        
        Returns:
            dict: Résultat de l'inscription avec success (bool) et message (str)
        """
        try:
            # Vérifier si l'email existe déjà
            if self.email_exists(email):
                return {"success": False, "message": "Cet email est déjà utilisé"}
            
            # Hacher le mot de passe
            password_hash = self.hash_password(password)
            
            # Insérer l'utilisateur
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO users (nom, prenoms, email, telephone, password_hash, investir, cherche_investisseurs)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (nom, prenoms, email, telephone, password_hash, investir, cherche_investisseurs))
            
            user_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            return {"success": True, "message": "Inscription réussie !", "user_id": user_id}
            
        except Exception as e:
            return {"success": False, "message": f"Erreur lors de l'inscription: {str(e)}"}
    
    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Connecte un utilisateur
        
        Returns:
            dict: Résultat de la connexion avec success (bool), message (str) et user_data (dict)
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, nom, prenoms, email, password_hash, investir, cherche_investisseurs
                FROM users 
                WHERE email = %s
            """, (email,))
            
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if user is None:
                return {"success": False, "message": "Email ou mot de passe incorrect"}
            
            # Vérifier le mot de passe
            if self.verify_password(password, user[4]):
                user_data = {
                    "id": user[0],
                    "nom": user[1],
                    "prenoms": user[2],
                    "email": user[3],
                    "investir": user[5],
                    "cherche_investisseurs": user[6]
                }
                return {"success": True, "message": "Connexion réussie !", "user_data": user_data}
            else:
                return {"success": False, "message": "Email ou mot de passe incorrect"}
                
        except Exception as e:
            return {"success": False, "message": f"Erreur lors de la connexion: {str(e)}"}
    
    def email_exists(self, email: str) -> bool:
        """Vérifie si un email existe déjà"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return count > 0
            
        except Exception as e:
            st.error(f"Erreur lors de la vérification de l'email: {str(e)}")
            return False
    
    def get_user_stats(self) -> Dict[str, int]:
        """Récupère les statistiques des utilisateurs"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Compter les utilisateurs actifs
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]
            
            # Compter les investisseurs
            cursor.execute("SELECT COUNT(*) FROM users WHERE investir = TRUE")
            investisseurs = cursor.fetchone()[0]
            
            # Compter ceux qui cherchent des investisseurs
            cursor.execute("SELECT COUNT(*) FROM users WHERE cherche_investisseurs = TRUE")
            cherchent_investisseurs = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "total_users": total_users,
                "investisseurs": investisseurs,
                "cherchent_investisseurs": cherchent_investisseurs
            }
            
        except Exception as e:
            st.error(f"Erreur lors de la récupération des statistiques: {str(e)}")
            return {"total_users": 0, "investisseurs": 0, "cherchent_investisseurs": 0}