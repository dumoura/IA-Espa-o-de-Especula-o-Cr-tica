#!/bin/bash

# Script para fazer push dos arquivos para o GitHub
# Repositório: https://github.com/dumoura/IA-Espa-o-de-Especula-o-Cr-tica.git

echo "🚀 Iniciando push para o GitHub..."

# Navegar para o diretório do projeto
cd /Users/dumoura/DevCursor

# Verificar se já é um repositório Git
if [ ! -d ".git" ]; then
    echo "📁 Inicializando repositório Git..."
    git init
fi

# Adicionar todos os arquivos
echo "📝 Adicionando arquivos..."
git add .

# Configurar o repositório remoto
echo "🔗 Configurando repositório remoto..."
git remote add origin https://github.com/dumoura/IA-Espa-o-de-Especula-o-Cr-tica.git 2>/dev/null || git remote set-url origin https://github.com/dumoura/IA-Espa-o-de-Especula-o-Cr-tica.git

# Fazer commit
echo "💾 Fazendo commit..."
git commit -m "Initial commit: Projeto IA e Visões de Futuro - Creative Coding

- Documentação completa do projeto
- Espaços de Especulação Crítica
- Protocolos Indígenas e IA
- Conceito de algorhythms
- Recursos e referências organizados
- Estrutura para experimentos e dados"

# Push para o GitHub
echo "⬆️ Fazendo push para o GitHub..."
git push -u origin main

echo "✅ Push concluído com sucesso!"
echo "🌐 Repositório disponível em: https://github.com/dumoura/IA-Espa-o-de-Especula-o-Cr-tica"
