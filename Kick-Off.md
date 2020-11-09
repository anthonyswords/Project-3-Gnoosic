<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100"/>

# Welcome to the Gnoosic project!
# Music Recomender Software

**Toni Espadas**

**Ironhack Barcelona Oct 2020**

## Content
- [Project Description](#project-description)
- [Workflow](#workflow)

## Project Description

This software recommends songs that have similar features with the song that the user inputs. 
From a database of 10000 songs which has been splitted into 10 clusters based on the songs features, the software selects a random song from a cluster depending on the song features that the user inputs. 


## Software Workflow

1. User inputs a song
2. If the song is currently on the current Billboard top 100 list, another song from the Billboard top 100 list is returned. 
3. If the song is not on the current Billboard top 100 list, the software gets the song features from the song inputed by the user from Spotify, and returns a song with similar features from the database cluster we have created. 

