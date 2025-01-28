(defproject spotify-stats "0.1.0-SNAPSHOT"
  :description "Spotify statistics analyzer"
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [clj-http "3.12.3"]
                 [cheshire "5.11.0"]
                 [environ "1.2.0"]]
  :main ^:skip-aot spotify-stats.core
  :target-path "data/raw"
  :profiles {:uberjar {:aot :all}
             :dev {:env {:spotify-client-id "your_client_id_here"
                        :spotify-client-secret "your_client_secret_here"}}})