# Где управлять моими сайтами

Проверка публичных DNS и хостинга: 13 ноября 2026.

| Сайт | DNS | Публикация главных страниц | Рабочий сервис Cloudflare |
|---|---|---|---|
| bmychance.com | GoDaddy | Vercel | bmychance-accounts |
| btc2g.com | Namecheap | GitHub Pages (vitalydz.github.io) | btc2g-accounts |
| vitdart.com | Namecheap | Vercel | vitdart-accounts |
| bit2g.com | Namecheap | Vercel | bit2g-ideas |
| findbbest.com | Namecheap | Vercel | findbbest-accounts |
| aikickai.com | Hostinger | Vercel | aikickai-accounts |

## Куда заходить

- **Опубликовать главную страницу** — [Vercel](https://vercel.com/dashboard). Для BTC2G — [GitHub](https://github.com/vitalydz), настройки Pages нужного репозитория.
- **Изменить DNS** — [GoDaddy](https://dcc.godaddy.com/), [Namecheap](https://ap.www.namecheap.com/domains/list/) или [Hostinger](https://hpanel.hostinger.com/), по строке сайта в таблице. DNS и место покупки домена могут различаться.
- **Аккаунты, API, игры и базы** — [Cloudflare](https://dash.cloudflare.com/). Код серверов в Workers & Pages, базы отдельно в Storage & databases → D1. Пустой раздел Domains не мешает работе этих сервисов через workers.dev.

## Тестовые сервисы BMyChance

| Worker | Назначение |
|---|---|
| bmychance-two-games-review | Предпросмотр игр, включая бильярд |
| bmychance-photo-review-preview | Предпросмотр проверки фотографий |

В панели было 11 Workers. Памятка перечисляет известные рабочие сервисы шести сайтов и две видимые тестовые версии. Остальные записи требуют отдельной сверки. Название Worker в таблице помогает найти его в панели; рабочий трафик отдельных маршрутов этой проверкой не измерялся.
