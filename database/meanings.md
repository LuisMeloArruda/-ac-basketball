- playerID: as a string (!WARNING may also include coaches (e.g. award_players.csv))
- award: Name as a string
- year: Int, starts from 1
- lgID: League ID name as a string (all "WNBA"!!)
- coachID: as a string
- tmID: Team ID name as a string
- stint: A stint is a unique period during which a player is continuously part of a team. If the player leaves and comes back, that’s a new stint.
- won: Games won in the season
- lost: Games lost in the season
- post_wins: Post-season wins
- post_losses: Post-season losses
- bioID: as a string
- pos: 'C' (Center), 'F' (forward) or 'G' (guard) (none, one or two)
- firstseason: (Always 0???)
- lastseason: (Always 0????)
- height: in inches
- weight: in pounds
- college: Name in string
- collegeOther: Name in string, may be empty (less important???)
- birthDate: YYYY-MM-DD
- deathDate: YYYY-MM-DD or 0000-00-00 if null
- GP: Games played as uint
- GS: Games started (not substitute/benched player)
- minutes: Minutes played as uint
- points: as uint
- oRebounds: offensive rebounds
- dRebounds: defensive rebounds
- rebounds: soma dos ofensive e defense rebounds
- assists: as uint
- steals: as uint
- blocks: as uint
- turnovers: as uint
- PF: Personal Fouls
- fgAttempted: field goals attempted
- fgMade: field goals made
- ftAttempted: Free throws attempted
- ftMade: Free throws made
- threeAttempted: Three-pointers attempted
- threeMade: Three-pointers made
- dq: disqualifications
- PostGP: Post-season games played
- PostGS: Post-season games started
- Post: Post-season
- franchID: ???? O identificador da franquia de basquete ????
- confID: Conference ID ("EA" or "WE")
- divID: All empty!!!!
- rank: Placement of the teams in the year
- playoff: Team made playoffs? ("Y" or "N")
- seeded: Always 0!!!
- firstRound: Empty if not made in, else "W" or "L"
- semis: Empty if not made in, else "W" or "L"
- finals: Empty if not made in, else "W" or "L"
- name: team name as a string
- min: Minutos jogados no total da temporada
- attend: Total de público (torcedores) ao longo da temporada
- arena: noem da arena

> series_post.csv and teams_post.csv is self-explanatory

### What to remove from each table
> players_team
    - lgID (always "WNBA")

> players
    - collegeOther (too many nulls)
    - firstseason (always 0)
    - lastseason (always 0)
    - deathDate (not important)
    - birthDate (not important)

> awards_players
    - lgID (always "WNBA")

> series_post
    - lgIDWinner (alwats "WNBA")
    - lgIDLoser (always "WNBA")

> teams_post
    - lgID (always "WNBA")

> teams
    - lgID (always "WNBA")
    - divID (empty)
    - seeded (always 0)
    - tmORB (always 0)
    - tmDRB (always 0)
    - tmTRB (always 0)
    - opptmORB (always 0)
    - opptmDRB (always 0)
    - opptmTRB (always 0)

> coaches