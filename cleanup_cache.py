import config
from google.genai import Client

def main():
    if not config.GOOGLE_API_KEY:
        print("❌ Nincs API kulcs megadva.")
        return

    client = Client(api_key=config.GOOGLE_API_KEY)
    print("🧹 Aktív Context Cache-ek keresése és törlése...")
    
    try:
        active_caches = client.caches.list()
        count = 0
        for c in active_caches:
            print(f"🗑️ Törlés: {c.display_name} ({c.name})")
            client.caches.delete(name=c.name)
            count += 1
        print(f"✅ Kész. {count} cache törölve.")
    except Exception as e:
        print(f"⚠️ Hiba a takarítás során: {e}")

if __name__ == "__main__":
    main()