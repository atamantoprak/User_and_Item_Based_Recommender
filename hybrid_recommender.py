###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# ID'si verilen kullanıcı için item-based ve user-based recommender
# yöntemlerini kullanarak 10 film önerisi yapınız.

# Veri Seti Hikayesi

# Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler ile birlikte bu filmlere yapılan
# derecelendirme puanlarını barındırmaktadır. 27.278 filmde 2.000.0263 derecelendirme içermektedir. Bu veri seti ise 17 Ekim 2016
# tarihinde oluşturulmuştur. 138.493 kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasında verileri içermektedir. Kullanıcılar
# rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.

# Değişkenler

    # movie.csv
# movieId: Eşsiz film numarası.
# title  : Film adı
# genres : Tür

    # raiting.csv
# userid Eşsiz kullanıcı numarası. (UniqueID)
# movieId Eşsiz film numarası. (UniqueID)
# rating Kullanıcı tarafından filme verilen puan
# timestamp Değerlendirme tarihi

#########################################
# User Based Recommendation
#########################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Adım 1: movie, rating veri setlerini okutunuz
# Adım 2: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
# Adım 3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.
# Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz.

movie = pd.read_csv("Hybrid_Recommender/datasets/movie.csv")
raiting = pd.read_csv("Hybrid_Recommender/datasets/rating.csv")
df = movie.merge(raiting, how="left", on="movieId")
rate_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = rate_counts[rate_counts["title"] < 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()

#  Rastgele bir kullanıcı id’si seçiniz.
random_user = 108170
#random_user = int(pd.Series(user_movie_df.index).sample(1).values[0])
# referans kullanıcının indexine ulaşıp ardından sadece izlediği filmleri tutan bir liste oluşturulur
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notnull().any()].tolist()

# listedeki filmlerin izlenme doğrulaması
user_movie_df.loc[user_movie_df.index == random_user,user_movie_df.columns == "Willow (1988)"]

# sadece izlenen filmlerden oluşan tüm kulkaanıcıları içeren dataframe oluşturulur
movies_watched_df = user_movie_df[movies_watched]

# kullanıcıların bu filmlerden kaç tanesini izlediği öğrenilir
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()

# filmlerden %60 dan fazlasını izleyen kullanıcıların id lerini içeren bir seri oluşturulur
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.sort_values(by="movie_count", ascending=False)
perc = len(movies_watched) * 60 / 100
user_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
user_same_movies

# dataframe bu seri ile filtrelenir
final_df = movies_watched_df[movies_watched_df.index.isin(user_same_movies)]
final_df.head()

# final_df kullanıcılarının birbirleriyle olan korelasyonları hesaplanır
corr_df = final_df.T.corr().unstack().sort_values()
corr_df.head()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# referans kullanıcı ile korelasyonu 0.65 üzeri olan kullanıcılaar getirilir
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", 'corr']].reset_index(drop=True)
top_users = top_users.sort_values(by="corr",ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
raiting = pd.read_csv("Hybrid_Recommender/datasets/rating.csv")
top_users_raiting = top_users.merge(raiting[["userId", "movieId", "rating"]], how="inner")
top_users_raiting.head(20)
top_users_raiting = top_users_raiting[top_users_raiting["userId"] != random_user]

# korelasyon ve rating tek başına yeterli bilgi sağlamadığı için ikisi çarpılır
top_users_raiting["weighted_raiting"] = top_users_raiting["corr"] * top_users_raiting["rating"]

# weighted_raiting filmlere göre gruplanır ver ortalaması alınır
recommendation_df = top_users_raiting.groupby("movieId").agg({"weighted_raiting": "mean"})

# weighted_raiting ortalaması 3.5 olan  filmlerin ID leri getirilir
to_be_recommend = recommendation_df[recommendation_df["weighted_raiting"] >= 3.5].sort_values(by="weighted_raiting", ascending=False)
to_be_recommend.reset_index(inplace=True)
movie = pd.read_csv("Hybrid_Recommender/datasets/movie.csv")

# önerilecek filmler dataframei ve movie dataframei birleştirilerek filmlerin isimleri alınır
to_be_recommend = to_be_recommend.merge(movie[["movieId", 'title']])["title"][0:5]

#########################################
# User_based_Recommender Script
#########################################
def user_based_recommender(random_user,perc_rate, corr_th, score):
    """

    :param random_user:
    When creating recommendations, the user's userID to be used as a reference should be entered.
    :param perc_rate:
    Filtering users based on the viewing rate of the movies watched by the reference user.
    (Enter the percentage as a decimal)
    :param corr_th:
    The correlation ratio of users with the reference user.(Enter the percentage as a decimal)
    :param score:
    weighted raiting score
    :return:
     The recommended movies are returned.
    """
    import pandas as pd
    def create_user_movie_df():
        movie = pd.read_csv("Hybrid_Recommender/datasets/movie.csv")
        raiting = pd.read_csv("Hybrid_Recommender/datasets/rating.csv")
        df = movie.merge(raiting, how="left", on="movieId")
        rate_counts = pd.DataFrame(df["title"].value_counts())
        rare_movies = rate_counts[rate_counts["title"] < 1000].index
        common_movies = df[~df["title"].isin(rare_movies)]
        user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
        return user_movie_df
    user_movie_df = create_user_movie_df()
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notnull().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    user_movie_count.sort_values(by="movie_count", ascending=False)
    perc = len(movies_watched) * perc_rate
    user_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
    final_df = movies_watched_df[movies_watched_df.index.isin(user_same_movies)]
    corr_df = final_df.T.corr().unstack().sort_values()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >=corr_th)][
        ["user_id_2", 'corr']].reset_index(drop=True)
    top_users = top_users.sort_values(by="corr", ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    raiting = pd.read_csv("Hybrid_Recommender/datasets/rating.csv")
    top_users_raiting = top_users.merge(raiting[["userId", "movieId", "rating"]], how="inner")
    top_users_raiting = top_users_raiting[top_users_raiting["userId"] != random_user]
    top_users_raiting["weighted_raiting"] = top_users_raiting["corr"] * top_users_raiting["rating"]
    recommendation_df = top_users_raiting.groupby("movieId").agg({"weighted_raiting": "mean"})
    to_be_recommend = recommendation_df[recommendation_df["weighted_raiting"] >= score].sort_values(by="weighted_raiting",                                                                                                  ascending=False)
    to_be_recommend.reset_index(inplace=True)
    movie = pd.read_csv("Hybrid_Recommender/datasets/movie.csv")
    to_be_recommend = to_be_recommend.merge(movie[["movieId", 'title']])["title"][0:5]
    return to_be_recommend
user_based_recommender(108170,0.6,0.65,3.5)
#########################################
# Item Based Recommendation
#########################################

movie = pd.read_csv("Hybrid_Recommender/datasets/movie.csv")
raiting = pd.read_csv("Hybrid_Recommender/datasets/rating.csv")
df = movie.merge(raiting, how="left", on="movieId")
df.head()

choosen_movieid = raiting[(raiting["userId"] == random_user) & (raiting["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

choosen_title = movie[movie["movieId"] == choosen_movieid]["title"].values[0]

movie_name = user_movie_df[choosen_title]

item_based_movies = user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(6)
item_based_movies = item_based_movies[1:6].index

#########################################
# Item_based_Recommender Script
#########################################
def item_based_recommender(random_user):
    import pandas as pd
    def create_user_movie_df():
        movie = pd.read_csv("Hybrid_Recommender/datasets/movie.csv")
        raiting = pd.read_csv("Hybrid_Recommender/datasets/rating.csv")
        df = movie.merge(raiting, how="left", on="movieId")
        rate_counts = pd.DataFrame(df["title"].value_counts())
        rare_movies = rate_counts[rate_counts["title"] < 1000].index
        common_movies = df[~df["title"].isin(rare_movies)]
        user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
        return user_movie_df
    movie = pd.read_csv("Hybrid_Recommender/datasets/movie.csv")
    raiting = pd.read_csv("Hybrid_Recommender/datasets/rating.csv")
    df = movie.merge(raiting, how="left", on="movieId")
    choosen_movieid = \
    raiting[(raiting["userId"] == random_user) & (raiting["rating"] == 5.0)].sort_values(by="timestamp",
                                                                                         ascending=False)["movieId"][0:1].values[0]
    choosen_title = movie[movie["movieId"] == choosen_movieid]["title"].values[0]
    movie_name = user_movie_df[choosen_title]
    item_based_movies = user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(6)
    item_based_movies = item_based_movies[1:6].index
    return item_based_movies
item_based_recommender(108170)