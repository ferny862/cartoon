# Stick Ops

A first-person 3D stick-figure shooter styled after modern Call of Duty
multiplayer: Team Deathmatch against bots, 6 v 6 up to 12 v 12, on four maps.

Open `index.html` in a desktop browser with hardware acceleration on. It loads
Three.js from cdnjs, so it needs an internet connection. It is played with a
keyboard and mouse; the mouse is captured when you deploy, and Esc releases it.

`classic.html` is the original 2D side-scrolling version.

## What's in it

- **Maps:**
  - **Downtown:** a burning city block at dusk with a central tower, a house
    roof and a sniper platform.
  - **Dockyard:** a rainy container port with stacked containers, a crane and
    a warehouse roof.
  - **Outpost:** a sun-baked desert compound with flat rooftops and low walls.
  - **Shipment:** a tiny container yard at sunset built for close-range chaos.
- **Lobby size:** 6 v 6 (first to 75), 9 v 9 (first to 100) or 12 v 12 (first
  to 125).
- **Weapons:** M4A1 and AK-47 (assault rifles), MP5 and Vector (SMGs), Model
  680 (shotgun), KAR98K (sniper), PKM (LMG), SKS (marksman), plus the X12 and
  .50 GS sidearms and frag grenades you can cook.
- **Gunsmith and classes:** build three custom classes. Each gun takes five
  attachments that change both its stats and its look:
  - **Optic:** iron sights, red dot, holographic, 3x tactical; snipers get 8x
    or 4x scopes.
  - **Muzzle:** flash hider, compensator or suppressor (keeps you off enemy
    radar).
  - **Barrel:** short or long.
  - **Underbarrel:** vertical grip, angled grip or tac laser.
  - **Magazine:** fast or extended.

  Add a camo (Woodland, Desert, Urban, Tiger, Gold, Diamond) and a perk
  (Scavenger, Lightweight, Ghost, Quick Fix, Hardline). The Gunsmith shows a
  3D preview and stat bars that turn green or red as you change parts.
- **Leveling:** new guns, attachments, camos and perks unlock as you level up
  (max level 30). The menu shows the next unlock, and level-ups during a match
  announce what you just earned.
- **Changing guns mid-match:** press B (or Esc) to pick another class. It
  equips right away if you spawned in the last 10 seconds, otherwise on your
  next spawn. Fallen soldiers drop their guns: press E to swap for one, or to
  take ammo from the same gun.
- **First-person feel:** aim down sights with field-of-view zoom, recoil, weapon
  bob and sway, tactical sprint, slide, crouch and jump.
- **HUD:** rotating minimap with red pings when enemies fire, compass bar, team
  score bars and match clock, kill feed, ammo counter, hitmarkers (red on
  kills), damage direction arcs, grenade warnings, and a scoreboard on Tab.
- **Killcam:** after you die, you see the fight from your killer's eyes.
- **Killstreaks:** UAV (3 kills), Airstrike (5; aim where it should land and
  click) and Attack Helo (7). Enemy bots earn and call them too.
- **Medals:** First Blood, Headshot, Longshot, Double Kill and more, plus
  level-ups. XP is saved in the browser.
- **Bots:** they find paths around the map (including stairs to the rooftops),
  fight at their weapon's best range, flank, retreat when wounded, and throw
  grenades. Three difficulty levels.
- **Audio:** sound effects are generated in the browser and positioned in 3D,
  with spoken announcements where the browser supports them.

## Controls

| Key | Action |
|---|---|
| W A S D | Move |
| Mouse | Look · left click fire · right click aim down sights |
| Space | Jump |
| Shift | Sprint |
| C / Ctrl | Crouch (slide while sprinting) |
| R | Reload |
| Q / mouse wheel | Swap weapon |
| G (hold) | Cook and throw frag |
| V / F | Melee |
| E | Pick up a dropped weapon |
| B | Change class |
| 3 / 4 / 5 | UAV · Airstrike · Attack helo |
| Tab | Scoreboard |
| M | Mute |
| Esc | Pause |
