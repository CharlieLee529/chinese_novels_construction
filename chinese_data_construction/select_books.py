from bs4 import BeautifulSoup
import json 

# Load the HTML content
with open('./Best Books Ever (123864 books).html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'lxml')

# Find all the book entries
book_entries = soup.find_all('tr', {'id': lambda x: x and x.startswith('review_')})

# Extract the data
books = []
for entry in book_entries:
    title_tag = entry.find('a', class_='bookTitle')
    title = title_tag.text.strip() if title_tag else 'N/A'
    url = 'https://www.goodreads.com' + title_tag['href'] if title_tag else 'N/A'
    
    author_tag = entry.find('a', class_='authorName')
    author = author_tag.text.strip() if author_tag else 'N/A'
    
    score_tag = entry.find('span', class_='minirating')
    score_text = score_tag.text.strip() if score_tag else 'N/A'
    score = float(score_text.split(' — ')[0].split()[0]) if score_tag else 'N/A'
    
    vote_tag = entry.find('span', class_='smallText uitext')
    vote_text = vote_tag.text.strip() if vote_tag else 'N/A'
    votes = int(vote_text.split()[0].replace(',', '')) if vote_tag else 'N/A'
    
    books.append({
        'title': title,
        'author': author,
        'score': score,
        'votes': votes,
        'url': url
    })

# Save the results to a JSON file
json_output_path = 'books_list.json'
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(books, f, ensure_ascii=False, indent=4)

print(f'Successfully saved {len(books)} books to {json_output_path}')