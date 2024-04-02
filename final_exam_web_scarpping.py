import requests
from bs4 import BeautifulSoup
import csv
import re

def scrape_web_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract the required information from the web page
    restaurant_name = soup.find("h1", class_="css-1se8maq").text.strip()
    total_reviews = soup.find("a", class_="css-19v1rkv").text.strip()
    rating = soup.find("span", class_="css-1fdy0l5").text.strip()

    reviews = []
    review_elements = soup.find_all("li", class_="css-1q2nwpv")
    
    for review_element in review_elements:
        review_text = review_element.find("p").text.strip()
        reviewer = review_element.find("span").text.strip()
        
        review = {
            "review_text": review_text,
            "reviewer": reviewer,
        }
        reviews.append(review)
    print(restaurant_name)
    print(total_reviews)
    print(reviews)
    print(rating)
    return restaurant_name, total_reviews, rating, reviews

def clean_data(data):
    # Clean the restaurant name
    cleaned_name = data[0].replace(",", "")
    
    # Clean the total reviews count
    cleaned_reviews = data[1].replace(" reviews", "").replace(",", "")
    clean_rating = data[2]
    # Clean each review
    cleaned_reviews_list = []
    for review in data[3]:
        cleaned_review_text = review["review_text"].replace(",", "")
        cleaned_reviewer = review["reviewer"].replace(",", "")
        #cleaned_rating = review["rating"].replace(" out of 5", "")
        
        cleaned_review = {
            "review_text": cleaned_review_text,
            "reviewer": cleaned_reviewer,
            #"rating": cleaned_rating
        }
        cleaned_reviews_list.append(cleaned_review)
    
    return cleaned_name, cleaned_reviews, clean_rating, cleaned_reviews_list

def write_to_csv(data, filename):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Restaurant Name", "Total Reviews", "Rating", "Review Text", "Reviewer"])
        
        for review in data[3]:
            writer.writerow([data[0], data[1], data[2],review["review_text"], review["reviewer"]])

def main():
    url = input("Enter the URL of the restaurant web page: ")
    filename = input("Enter the filename for the CSV file: ")
    
    # Scrape the web page and extract the required information
    scraped_data = scrape_web_page(url)
    
    # Clean the extracted data
    cleaned_data = clean_data(scraped_data)
    
    # Write the cleaned data to a CSV file
    write_to_csv(cleaned_data, filename)
    
    print("Data has been scraped, cleaned, and stored in the CSV file.")

if __name__ == "__main__":
    main()