(ns spotify-stats.core
  (:require [clj-http.client :as http]
            [cheshire.core :as json]
            [clojure.string :as str]
            [clojure.java.browse :as browse]
            [environ.core :refer [env]])
  (:gen-class))

(def client-id (env :spotify-client-id))
(def client-secret (env :spotify-client-secret))
(def redirect-uri "http://localhost:8080/callback")
(def auth-url "https://accounts.spotify.com/authorize")
(def token-url "https://accounts.spotify.com/api/token")

(def scopes ["user-top-read"
             "user-read-recently-played"
             "user-library-read"])

(defn create-auth-url []
  (str auth-url
       "?client_id=" client-id
       "&response_type=code"
       "&redirect_uri=" redirect-uri
       "&scope=" (str/join "%20" scopes)))

(defn get-access-token [code]
  (let [response (http/post token-url
                           {:form-params {:grant_type "authorization_code"
                                        :code code
                                        :redirect_uri redirect-uri
                                        :client_id client-id
                                        :client_secret client-secret}
                            :as :json})]
    (-> response :body)))

(defn spotify-get [endpoint access-token]
  (-> (http/get (str "https://api.spotify.com/v1" endpoint)
                {:headers {"Authorization" (str "Bearer " access-token)}
                 :as :json})
      :body))

(defn get-top-tracks [access-token time-range]
  (spotify-get (str "/me/top/tracks?time_range=" time-range "&limit=50")
               access-token))

(defn extract-track-info [track]
  {:name (:name track)
   :artist (-> track :artists first :name)
   :popularity (:popularity track)
   :duration_ms (:duration_ms track)})

(defn analyze-top-tracks [tracks]
  (->> tracks
       :items
       (map extract-track-info)
       (map-indexed #(assoc %2 :rank (inc %1)))))

(defn analyze-recent-tracks [tracks]
  (->> tracks
       :items
       (map (fn [item]
              (merge
               (extract-track-info (:track item))
               {:played_at (:played_at item)})))))

(defn get-recently-played [access-token]
  (spotify-get "/me/player/recently-played?limit=50"  ; Maximum allowed by API
               access-token))

(defn save-to-csv [data filename]
  (with-open [writer (clojure.java.io/writer filename)]
    (let [columns (keys (first data))]
      ; Write header
      (.write writer (str (str/join "," (map name columns)) "\n"))
      ; Write data rows with proper escaping
      (doseq [row data]
        (.write writer 
          (str (str/join "," 
            (map #(let [value (str (get row %))]
                   (if (str/includes? value ",")
                     (str "\"" value "\"")  ; Quote fields containing commas
                     value))
                 columns))
               "\n"))))))

(defn get-full-history [access-token]
  (let [recent (get-recently-played access-token)
        top-short (get-top-tracks access-token "short_term")
        top-medium (get-top-tracks access-token "medium_term")
        top-long (get-top-tracks access-token "long_term")]
    
    ; Process and save data
    (save-to-csv (analyze-recent-tracks recent) "spotify_recent_tracks.csv")
    (save-to-csv (analyze-top-tracks top-short) "spotify_top_tracks_short.csv")
    (save-to-csv (analyze-top-tracks top-medium) "spotify_top_tracks_medium.csv")
    (save-to-csv (analyze-top-tracks top-long) "spotify_top_tracks_long.csv")
    
    ; Return processed data for further analysis
    {:recent recent
     :top-short top-short
     :top-medium top-medium
     :top-long top-long}))

(defn authenticate []
  (println "Opening browser for authentication...")
  (browse/browse-url (create-auth-url))
  (println "Please enter the code from the redirect URL:")
  (let [code (read-line)]
    (get-access-token code)))

(defn analyze-listening-habits []
  (let [token (:access_token (authenticate))
        top-tracks-short (analyze-top-tracks (get-top-tracks token "short_term"))
        top-tracks-medium (analyze-top-tracks (get-top-tracks token "medium_term"))
        top-tracks-long (analyze-top-tracks (get-top-tracks token "long_term"))
        recent-tracks (analyze-recent-tracks (get-recently-played token))]
    
    ; Save data to CSV files
    (save-to-csv top-tracks-short "spotify_top_tracks_short.csv")
    (save-to-csv top-tracks-medium "spotify_top_tracks_medium.csv")
    (save-to-csv top-tracks-long "spotify_top_tracks_long.csv")
    (save-to-csv recent-tracks "spotify_recent_tracks.csv")
    
    ; Print some basic analysis
    (println "\nTop Artists (Last 4 Weeks):")
    (->> top-tracks-short
         (group-by :artist)
         (map (fn [[k v]] [k (count v)]))
         (sort-by second >)
         (take 5)
         (map println))
    
    {:top-tracks {:short-term top-tracks-short
                 :medium-term top-tracks-medium
                 :long-term top-tracks-long}
     :recent-tracks recent-tracks}))

(defn -main [& args]
  (analyze-listening-habits)
  (System/exit 0))