(ns spotify-analysis.config
  (:require [environ.core :refer [env]]
            [dotenv :refer [env load-dotenv]]))

(defn load-config []
  ;; Load .env file
  (load-dotenv)
  
  ;; Return config map
  {:client-id (or (env :spotify-client-id)
                  (throw (ex-info "Missing SPOTIFY_CLIENT_ID" {})))
   :client-secret (or (env :spotify-client-secret)
                     (throw (ex-info "Missing SPOTIFY_CLIENT_SECRET" {})))
   :output-dir (or (env :output-dir) "data/raw")})
